"""Measure and edit triangle meshes (docs/three.md). Pure numpy; every edit returns a new mesh."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rokkur_studio.mesh.io import Mesh, MeshError

AXES = {"x": 0, "y": 1, "z": 2}


def measure(mesh: Mesh) -> dict[str, Any]:
    """Counts, bounding box, surface area, volume and whether the mesh is closed. Volume and
    the closed check are what a printer slicer cares about; both are stated, never guessed."""
    tris = mesh.triangles
    if len(tris) == 0:
        box = mesh.vertices.min(axis=0) if len(mesh.vertices) else np.zeros(3)
        return {"vertices": int(len(mesh.vertices)), "triangles": 0, "min": box.tolist(),
                "max": box.tolist(), "size": [0.0, 0.0, 0.0], "area": 0.0, "volume": 0.0,
                "closed": False, "degenerate": 0, "open_edges": 0, "point_cloud": True}
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)
    area = float(double_area.sum() / 2)
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
    degenerate = int((double_area < 1e-12).sum())
    edges = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = int((counts != 2).sum())
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    return {"vertices": int(len(mesh.vertices)), "triangles": int(len(mesh.faces)),
            "min": lo.round(4).tolist(), "max": hi.round(4).tolist(),
            "size": (hi - lo).round(4).tolist(), "area": round(area, 4),
            "volume": round(abs(volume), 4), "inverted": volume < 0,
            "closed": open_edges == 0 and degenerate == 0, "degenerate": degenerate,
            "open_edges": open_edges, "point_cloud": False}


def transform(mesh: Mesh, matrix: np.ndarray) -> Mesh:
    out = mesh.copy()
    out.vertices = mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3]
    if np.linalg.det(matrix[:3, :3]) < 0:  # a mirror turns the triangles inside out
        out.faces = out.faces[:, [0, 2, 1]]
    return out


def scale(mesh: Mesh, factor: float | tuple[float, float, float]) -> Mesh:
    f = (factor, factor, factor) if isinstance(factor, int | float) else factor
    if any(v <= 0 for v in f):
        raise MeshError("scale factors must be positive")
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = f
    return transform(mesh, m)


def fit(mesh: Mesh, size: float, axis: str = "max") -> Mesh:
    """Scale uniformly so the longest side (or one axis) measures ``size``."""
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    extent = hi - lo
    current = float(extent.max()) if axis == "max" else float(extent[AXES[axis]])
    if current <= 0:
        raise MeshError("the mesh has no extent to fit")
    return scale(mesh, size / current)


def translate(mesh: Mesh, offset: tuple[float, float, float]) -> Mesh:
    m = np.eye(4)
    m[:3, 3] = offset
    return transform(mesh, m)


def center(mesh: Mesh, *, on_floor: bool = False) -> Mesh:
    """Put the bounding-box centre at the origin; ``on_floor`` keeps it standing on z = 0."""
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    mid = (lo + hi) / 2
    offset = [-mid[0], -mid[1], -lo[2] if on_floor else -mid[2]]
    return translate(mesh, (offset[0], offset[1], offset[2]))


def rotate(mesh: Mesh, axis: str, degrees: float) -> Mesh:
    i = AXES[axis]
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    r = np.eye(3)
    j, k = (i + 1) % 3, (i + 2) % 3
    r[j, j], r[j, k], r[k, j], r[k, k] = c, -s, s, c
    m = np.eye(4)
    m[:3, :3] = r
    return transform(mesh, m)


def mirror(mesh: Mesh, axis: str) -> Mesh:
    m = np.eye(4)
    m[AXES[axis], AXES[axis]] = -1
    return transform(mesh, m)


def flip_normals(mesh: Mesh) -> Mesh:
    out = mesh.copy()
    out.faces = out.faces[:, [0, 2, 1]]
    return out


def weld(mesh: Mesh, tolerance: float = 1e-6) -> Mesh:
    """Merge vertices closer than ``tolerance`` and drop the triangles that collapse."""
    if tolerance <= 0:
        raise MeshError("tolerance must be positive")
    keys = np.round(mesh.vertices / tolerance).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3))
    np.add.at(sums, inverse, mesh.vertices)
    counts = np.bincount(inverse, minlength=len(unique)).reshape(-1, 1)
    vertices = sums / counts
    faces = inverse[mesh.faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    out = Mesh(vertices, faces[keep], mesh.name, dict(mesh.meta))
    return drop_unused(out)


def drop_degenerate(mesh: Mesh) -> Mesh:
    tris = mesh.triangles
    area2 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    return drop_unused(Mesh(mesh.vertices, mesh.faces[area2 >= 1e-12], mesh.name, dict(mesh.meta)))


def drop_unused(mesh: Mesh) -> Mesh:
    used = np.unique(mesh.faces) if len(mesh.faces) else np.arange(0)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.vertices[used], remap[mesh.faces] if len(mesh.faces) else mesh.faces,
                mesh.name, dict(mesh.meta))


def combine(meshes: list[Mesh], name: str = "combined") -> Mesh:
    if len(meshes) < 2:
        raise MeshError("combining needs two or more meshes")
    verts, faces, base = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces + base)
        base += len(m.vertices)
    return Mesh(np.concatenate(verts), np.concatenate(faces), name)


def box(size: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Mesh:
    """A closed box, for tests and as a fixture."""
    x, y, z = size
    v = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0], [0, 0, z], [x, 0, z], [x, y, z],
                  [0, y, z]], dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6],
                  [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return Mesh(v, f, "box")


def heightfield(heights: np.ndarray, *, cell: float, base: float = 0.0,
                valid: np.ndarray | None = None) -> Mesh:
    """A closed, printable solid from a grid of heights (row 0 at the top, ``cell`` units per
    grid step): the top surface, four walls and a flat bottom at ``z = 0``. Heights are added
    to ``base``; cells where ``valid`` is False are dropped from the top and walled instead."""
    h = np.asarray(heights, dtype=np.float64)
    rows, cols = h.shape
    if rows < 2 or cols < 2:
        raise MeshError("a heightfield needs at least 2 x 2 samples")
    ok = np.ones_like(h, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    z = np.where(ok, h + base, 0.0)
    ys, xs = np.mgrid[0:rows, 0:cols]
    top = np.stack([xs * cell, (rows - 1 - ys) * cell, z], axis=-1).reshape(-1, 3)
    bottom = top.copy()
    bottom[:, 2] = 0.0
    n = rows * cols
    vid = np.arange(n).reshape(rows, cols)
    a, b = vid[:-1, :-1], vid[:-1, 1:]
    c, d = vid[1:, :-1], vid[1:, 1:]
    quad_ok = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    a, b, c, d = a[quad_ok], b[quad_ok], c[quad_ok], d[quad_ok]
    top_faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    bottom_faces = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]) + n
    # walls along every edge that has a top quad on one side only
    edges = np.concatenate([np.stack([a, c], 1), np.stack([c, d], 1), np.stack([d, b], 1),
                            np.stack([b, a], 1)])
    key = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    border = edges[counts[inverse] == 1]
    p, q = border[:, 0], border[:, 1]
    walls = np.concatenate([np.stack([p, q + n, q], 1), np.stack([p, p + n, q + n], 1)])
    faces = np.concatenate([top_faces, bottom_faces, walls])
    return drop_unused(Mesh(np.concatenate([top, bottom]), faces, "heightfield"))


def relief_from_gray(gray: np.ndarray, *, width_mm: float, depth_mm: float, base_mm: float,
                     invert: bool = False) -> Mesh:
    """A relief or lithophane from a grey picture: bright is high (or low with ``invert``,
    which is what a backlit lithophane wants: thin where light should pass)."""
    g = np.asarray(gray, dtype=np.float64) / 255.0
    if invert:
        g = 1.0 - g
    rows, cols = g.shape
    cell = width_mm / max(1, cols - 1)
    return heightfield(g * depth_mm, cell=cell, base=base_mm)


def relief_from_points(points: np.ndarray, *, cell: float, base: float = 1.0,
                       fill: int = 1) -> Mesh:
    """A 2.5D solid from a point cloud seen from above (terrain, a wall, a relief scan): the
    highest point per grid cell becomes the surface. Holes up to ``fill`` cells wide are
    filled from their neighbours; overhangs and the back of an object cannot be recovered
    this way (that needs a full surface reconstruction such as Poisson)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise MeshError("too few points")
    if cell <= 0:
        raise MeshError("the cell size must be positive")
    lo = pts.min(axis=0)
    ij = np.floor((pts[:, :2] - lo[:2]) / cell).astype(np.int64)
    cols, rows = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    if rows * cols > 4_000_000:
        raise MeshError(f"{cols} x {rows} cells is too fine; use a larger cell size")
    grid = np.full((rows, cols), -np.inf)
    np.maximum.at(grid, (rows - 1 - ij[:, 1], ij[:, 0]), pts[:, 2] - lo[2])
    for _ in range(max(0, fill)):
        empty = ~np.isfinite(grid)
        if not empty.any():
            break
        padded = np.pad(grid, 1, constant_values=-np.inf)
        neighbours = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2],
                               padded[1:-1, 2:]])
        best = neighbours.max(axis=0)
        grid = np.where(empty & np.isfinite(best), best, grid)
    valid = np.isfinite(grid)
    if rows < 2 or cols < 2 or valid.sum() < 4:
        raise MeshError("the points cover too few cells; use a smaller cell size")
    mesh = heightfield(np.where(valid, grid, 0.0), cell=cell, base=base, valid=valid)
    return translate(mesh, (float(lo[0]), float(lo[1]), 0.0))


def preview_svg(mesh: Mesh, *, width: int = 640, height: int = 480, max_faces: int = 20000
                ) -> str:
    """A shaded isometric drawing of the mesh as SVG: what a page shows without WebGL."""
    if len(mesh.faces) == 0:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
                f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" fill="#888">'
                "no triangles</text></svg>")
    faces = mesh.faces
    if len(faces) > max_faces:
        faces = faces[np.linspace(0, len(faces) - 1, max_faces).astype(int)]
    m = rotate(rotate(Mesh(mesh.vertices, faces), "x", -60), "z", 35)
    tris = m.triangles
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 0)
    front = normals[:, 2] > 0
    tris, normals = tris[front], normals[front]
    if len(tris) == 0:
        tris, normals = m.triangles, -np.cross(m.triangles[:, 1] - m.triangles[:, 0],
                                               m.triangles[:, 2] - m.triangles[:, 0])
    order = np.argsort(tris[:, :, 2].mean(axis=1))
    tris, normals = tris[order], normals[order]
    lo, hi = m.vertices.min(axis=0), m.vertices.max(axis=0)
    span = max(float((hi - lo)[:2].max()), 1e-9)
    s = min(width, height) * 0.86 / span
    ox = width / 2 - (lo[0] + hi[0]) / 2 * s
    oy = height / 2 + (lo[1] + hi[1]) / 2 * s
    light = np.array([0.3, 0.5, 0.8])
    light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0, 1) * 0.65 + 0.3
    polys = []
    for t, k in zip(tris, shade, strict=True):
        pts = " ".join(f"{p[0] * s + ox:.1f},{oy - p[1] * s:.1f}" for p in t)
        c = int(60 + 170 * k)
        polys.append(f'<polygon points="{pts}" fill="rgb({c},{int(c * 0.78)},{int(c * 0.45)})"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}"><rect width="100%" height="100%" fill="#13141c"/>'
            + "".join(polys) + "</svg>")
