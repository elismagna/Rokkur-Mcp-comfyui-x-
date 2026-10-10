"""Read and write triangle meshes without any library beyond numpy (docs/three.md).

Formats: STL (binary and ASCII), OBJ (``v``/``f``, polygons fan-triangulated), PLY (ASCII and
binary little endian, vertex/face elements) and GLB read-only (glTF 2.0 binary with embedded
buffers, triangle primitives). Everything is a ``Mesh`` of float64 vertices and int64 faces.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MESH_EXTS = frozenset({".stl", ".obj", ".ply", ".glb"})
POINT_EXTS = frozenset({".xyz", ".pts", ".txt", ".csv"})  # plain point lists (LiDAR exports)
WRITE_EXTS = ("stl", "obj", "ply", "glb")


class MeshError(ValueError):
    pass


@dataclass
class Mesh:
    vertices: np.ndarray                    # (n, 3) float64
    faces: np.ndarray                       # (m, 3) int64, indices into vertices
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if len(self.faces) and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise MeshError("a face points outside the vertex list")

    def copy(self) -> Mesh:
        return Mesh(self.vertices.copy(), self.faces.copy(), self.name, dict(self.meta))

    @property
    def triangles(self) -> np.ndarray:
        """(m, 3, 3): the corner points of every face."""
        return self.vertices[self.faces]


# -- STL ------------------------------------------------------------------------------------
def _read_stl(data: bytes, name: str) -> Mesh:
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + count * 50 and not data[:5].lower().startswith(b"solid"):
            return _read_stl_binary(data, count, name)
        if len(data) == 84 + count * 50:
            return _read_stl_binary(data, count, name)  # "solid" header but binary sized
    return _read_stl_ascii(data.decode("utf-8", "replace"), name)


def _read_stl_binary(data: bytes, count: int, name: str) -> Mesh:
    records = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                   ("attr", "<u2")]), count=count, offset=84)
    tris = records["v"].astype(np.float64)
    return _from_triangles(tris, name, {"format": "stl", "encoding": "binary"})


def _read_stl_ascii(text: str, name: str) -> Mesh:
    values = re.findall(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text)
    if not values or len(values) % 3:
        raise MeshError("not an STL file (no complete triangles found)")
    tris = np.array(values, dtype=np.float64).reshape(-1, 3, 3)
    return _from_triangles(tris, name, {"format": "stl", "encoding": "ascii"})


def _from_triangles(tris: np.ndarray, name: str, meta: dict[str, Any]) -> Mesh:
    """Shared vertices from a triangle soup: identical points become one vertex."""
    flat = tris.reshape(-1, 3)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    return Mesh(unique, inverse.reshape(-1, 3), name, meta)


def _write_stl(mesh: Mesh, path: Path, *, ascii_: bool = False) -> None:
    tris = mesh.triangles
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    if ascii_:
        lines = [f"solid {mesh.name or 'rokkur'}"]
        for n, t in zip(normals, tris, strict=True):
            lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop")
            lines += [f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}" for p in t]
            lines.append("    endloop\n  endfacet")
        lines.append(f"endsolid {mesh.name or 'rokkur'}")
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return
    records = np.zeros(len(tris), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                   ("attr", "<u2")]))
    records["n"], records["v"] = normals, tris
    header = (f"Rokkur Studio {mesh.name}".encode()[:80]).ljust(80, b"\0")
    path.write_bytes(header + struct.pack("<I", len(tris)) + records.tobytes())


# -- OBJ ------------------------------------------------------------------------------------
def _read_obj(text: str, name: str) -> Mesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            idx = []
            for token in line.split()[1:]:
                value = int(token.split("/")[0])
                idx.append(value - 1 if value > 0 else len(vertices) + value)
            for i in range(1, len(idx) - 1):  # fan-triangulate polygons
                faces.append([idx[0], idx[i], idx[i + 1]])
    if not vertices:
        raise MeshError("not an OBJ file (no vertices)")
    return Mesh(np.array(vertices), np.array(faces, dtype=np.int64).reshape(-1, 3), name,
                {"format": "obj"})


def _write_obj(mesh: Mesh, path: Path) -> None:
    lines = [f"# Rokkur Studio {mesh.name}", f"o {mesh.name or 'mesh'}"]
    lines += [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in mesh.vertices]
    lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in mesh.faces]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# -- PLY ------------------------------------------------------------------------------------
_PLY_TYPES = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1", "short": "i2",
              "ushort": "u2", "int16": "i2", "uint16": "u2", "int": "i4", "uint": "u4",
              "int32": "i4", "uint32": "u4", "float": "f4", "float32": "f4", "double": "f8",
              "float64": "f8"}


def _read_ply(data: bytes, name: str) -> Mesh:
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise MeshError("not a PLY file")
    header = data[:end].decode("ascii", "replace").splitlines()
    body = data[end + len("end_header"):]
    body = body[1:] if body[:1] == b"\n" else body[2:] if body[:2] == b"\r\n" else body
    fmt = next((h.split()[1] for h in header if h.startswith("format")), "ascii")
    elements: list[tuple[str, int, list[tuple[str, str, str | None, str | None]]]] = []
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                elements[-1][2].append((parts[4], "list", parts[2], parts[3]))
            else:
                elements[-1][2].append((parts[2], parts[1], None, None))
    vertices = np.zeros((0, 3))
    faces: list[list[int]] = []
    if fmt == "ascii":
        tokens = body.decode("ascii", "replace").split()
        pos = 0
        for ename, count, props in elements:
            rows = []
            for _ in range(count):
                row: dict[str, Any] = {}
                for pname, ptype, _ct, _it in props:
                    if ptype == "list":
                        n = int(tokens[pos])
                        row[pname] = [int(t) for t in tokens[pos + 1:pos + 1 + n]]
                        pos += 1 + n
                    else:
                        row[pname] = float(tokens[pos])
                        pos += 1
                rows.append(row)
            if ename == "vertex":
                vertices = np.array([[r["x"], r["y"], r["z"]] for r in rows])
            elif ename == "face":
                key = next((p[0] for p in props if p[1] == "list"), "vertex_indices")
                for r in rows:
                    idx = r[key]
                    faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    else:
        order = "<" if fmt == "binary_little_endian" else ">"
        pos = 0
        for ename, count, props in elements:
            if all(p[1] != "list" for p in props):
                dtype = np.dtype([(p[0], order + _PLY_TYPES[p[1]]) for p in props])
                arr = np.frombuffer(body, dtype=dtype, count=count, offset=pos)
                pos += count * dtype.itemsize
                if ename == "vertex":
                    vertices = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
                continue
            for _ in range(count):
                row = {}
                for pname, ptype, ctype, itype in props:
                    if ptype == "list":
                        cdt = np.dtype(order + _PLY_TYPES[str(ctype)])
                        n = int(np.frombuffer(body, dtype=cdt, count=1, offset=pos)[0])
                        pos += cdt.itemsize
                        idt = np.dtype(order + _PLY_TYPES[str(itype)])
                        row[pname] = np.frombuffer(body, dtype=idt, count=n, offset=pos).tolist()
                        pos += n * idt.itemsize
                    else:
                        dt = np.dtype(order + _PLY_TYPES[ptype])
                        row[pname] = float(np.frombuffer(body, dtype=dt, count=1, offset=pos)[0])
                        pos += dt.itemsize
                if ename == "face":
                    idx = row.get("vertex_indices", row.get("vertex_index", []))
                    faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    if not len(vertices):
        raise MeshError("the PLY file has no vertices")
    return Mesh(vertices, np.array(faces, dtype=np.int64).reshape(-1, 3), name,
                {"format": "ply", "encoding": fmt, "point_cloud": not faces})


def _write_ply(mesh: Mesh, path: Path) -> None:
    header = ("ply\nformat binary_little_endian 1.0\ncomment Rokkur Studio\n"
              f"element vertex {len(mesh.vertices)}\nproperty float x\nproperty float y\n"
              f"property float z\nelement face {len(mesh.faces)}\n"
              "property list uchar int vertex_indices\nend_header\n")
    verts = mesh.vertices.astype("<f4").tobytes()
    face_dtype = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    faces = np.zeros(len(mesh.faces), dtype=face_dtype)
    faces["n"], faces["i"] = 3, mesh.faces
    path.write_bytes(header.encode("ascii") + verts + faces.tobytes())


# -- GLB (read) ------------------------------------------------------------------------------
_GL_TYPES = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_GL_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_glb(data: bytes, name: str) -> Mesh:
    if data[:4] != b"glTF":
        raise MeshError("not a GLB file")
    length = struct.unpack("<I", data[8:12])[0]
    pos, doc, blob = 12, None, b""
    while pos < min(length, len(data)):
        clen, ctype = struct.unpack("<II", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + clen]
        if ctype == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            blob = chunk
        pos += 8 + clen
    if doc is None:
        raise MeshError("the GLB file has no JSON chunk")

    def accessor(index: int) -> np.ndarray:
        acc = doc["accessors"][index]
        view = doc["bufferViews"][acc["bufferView"]]
        dtype = np.dtype("<" + _GL_TYPES[acc["componentType"]])
        width = _GL_SIZES[acc["type"]]
        offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride")
        if stride and stride != dtype.itemsize * width:
            rows = np.frombuffer(blob, dtype=np.uint8, count=stride * acc["count"], offset=offset)
            rows = rows.reshape(acc["count"], stride)[:, :dtype.itemsize * width]
            return np.ascontiguousarray(rows).view(dtype).reshape(acc["count"], width)
        return np.frombuffer(blob, dtype=dtype, count=acc["count"] * width,
                             offset=offset).reshape(acc["count"], width)

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    base = 0
    for m in doc.get("meshes", []):
        for prim in m.get("primitives", []):
            if prim.get("mode", 4) != 4 or "POSITION" not in prim.get("attributes", {}):
                continue
            v = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
            if "indices" in prim:
                f = accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
            else:
                f = np.arange(len(v), dtype=np.int64).reshape(-1, 3)
            all_v.append(v)
            all_f.append(f + base)
            base += len(v)
    if not all_v:
        raise MeshError("the GLB file has no triangle mesh")
    return Mesh(np.concatenate(all_v), np.concatenate(all_f), name, {"format": "glb"})


def glb_bytes(mesh: Mesh) -> bytes:
    """A minimal glTF 2.0 binary: one mesh, one triangle primitive, positions and indices."""
    pos = mesh.vertices.astype("<f4")
    idx = mesh.faces.astype("<u4")
    pos_bytes, idx_bytes = pos.tobytes(), idx.tobytes()
    pad = (-len(pos_bytes)) % 4
    blob = pos_bytes + b"\0" * pad + idx_bytes
    blob += b"\0" * ((-len(blob)) % 4)
    lo = pos.min(axis=0).tolist() if len(pos) else [0, 0, 0]
    hi = pos.max(axis=0).tolist() if len(pos) else [0, 0, 0]
    doc = {"asset": {"version": "2.0", "generator": "Rokkur Studio"},
           "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"name": mesh.name or "mesh", "primitives": [
               {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
           "buffers": [{"byteLength": len(blob)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes),
                            "target": 34962},
                           {"buffer": 0, "byteOffset": len(pos_bytes) + pad,
                            "byteLength": len(idx_bytes), "target": 34963}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": int(len(pos)),
                          "type": "VEC3", "min": lo, "max": hi},
                         {"bufferView": 1, "componentType": 5125, "count": int(idx.size),
                          "type": "SCALAR"}]}
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(blob), 0x004E4942) + blob)


def read_points(path: Path) -> np.ndarray:
    """``(n, 3)`` points from a plain point list (x y z per line, spaces or commas; extra
    columns such as colour or intensity are ignored), or from a mesh file's vertices."""
    path = Path(path)
    if path.suffix.lower() in MESH_EXTS:
        return read_mesh(path).vertices
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = re.split(r"[\s,;]+", line.strip())
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue  # a header line
    if len(rows) < 3:
        raise MeshError(f"{path.name}: no x y z points found")
    return np.array(rows, dtype=np.float64)


# -- entry points ---------------------------------------------------------------------------
def read_mesh(path: Path) -> Mesh:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in MESH_EXTS:
        raise MeshError(f"{path.name}: use an STL, OBJ, PLY or GLB file")
    data = path.read_bytes()
    if not data:
        raise MeshError(f"{path.name} is empty")
    name = path.stem
    if suffix == ".stl":
        return _read_stl(data, name)
    if suffix == ".obj":
        return _read_obj(data.decode("utf-8", "replace"), name)
    if suffix == ".ply":
        return _read_ply(data, name)
    return _read_glb(data, name)


def write_mesh(mesh: Mesh, path: Path, *, ascii_stl: bool = False) -> Path:
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in WRITE_EXTS:
        raise MeshError(f"the studio writes {', '.join(WRITE_EXTS)}; not {suffix!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == "stl":
        _write_stl(mesh, path, ascii_=ascii_stl)
    elif suffix == "obj":
        _write_obj(mesh, path)
    elif suffix == "glb":
        path.write_bytes(glb_bytes(mesh))
    else:
        _write_ply(mesh, path)
    return path
