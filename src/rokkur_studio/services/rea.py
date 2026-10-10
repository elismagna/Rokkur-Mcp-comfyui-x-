"""REA, "Reverse Engineer Anything", as a studio tool (docs/rea.md).

REA (github.com/morluto/rea, MIT; Elis's fork at elismagna/rea) is a CLI and MCP server that
inspects native binaries, JavaScript and Electron apps, .NET assemblies and websites, and
returns Evidence: observations, inferences and what stays unresolved. The studio runs the
real ``rea`` command on local files through the job queue, keeps every run's JSON output
under ``data/rea/<run id>/`` and shows it on the REA page. Nothing here fakes an analysis:
when ``rea`` is not installed where the worker runs, the page says so and how to install it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.config import Settings
from rokkur_studio.db.models import Job, ReaRun, utcnow
from rokkur_studio.jobs.queue import enqueue

Preset = Literal["analyze", "inspect", "search", "function", "decompile", "xrefs", "trace",
                 "instructions", "doctor", "providers", "capabilities"]

# What each preset runs, what it needs, and what it is for. ``cmd`` is REA's subcommand.
PRESETS: dict[str, dict[str, Any]] = {
    "analyze": {"cmd": "analyze", "target": True, "query": None, "provider": True, "snapshot": True,
                "label": "Analyse a program or app",
                "help": "The overview of a native program, a JavaScript/Electron app folder or "
                        "an .asar: what it is made of, with evidence. Directories and .asar "
                        "files use the static JavaScript workflow and need no engine."},
    "inspect": {"cmd": "inspect-artifact", "target": True, "query": None, "provider": False,
                "snapshot": False, "label": "Inspect an artifact",
                "help": "Identify an artifact (an .asar, a bundle, a binary) and what REA can "
                        "do with it, without a full analysis."},
    "search": {"cmd": "search", "target": True, "query": "Text to search for", "provider": True,
               "snapshot": True, "label": "Search inside",
               "help": "Strings, names and references matching a text."},
    "function": {"cmd": "function", "target": True, "query": "Function name or address",
                 "provider": True, "snapshot": True, "label": "Function dossier",
                 "help": "What a function does: callers, callees, strings, pseudocode summary."},
    "decompile": {"cmd": "decompile", "target": True, "query": "Function address (0x…)",
                  "provider": True, "snapshot": True, "label": "Decompile",
                  "help": "Pseudocode of one function (needs Ghidra, Hopper or IDA)."},
    "xrefs": {"cmd": "xrefs", "target": True, "query": "Address (0x…)", "provider": True,
              "snapshot": True, "label": "Cross references",
              "help": "Who references an address, and what it references."},
    "trace": {"cmd": "trace", "target": True, "query": "Text to trace", "provider": True,
              "snapshot": True, "label": "Trace a feature",
              "help": "Follow a string or name through the program to the code that uses it."},
    "instructions": {"cmd": "instructions", "target": True, "query": "Address (0x…)",
                     "provider": True, "snapshot": True, "label": "Assembly instructions",
                     "help": "The instructions at an address, without pseudocode."},
    "doctor": {"cmd": "doctor", "target": False, "query": None, "provider": True, "snapshot": False,
               "label": "Doctor: is REA ready?",
               "help": "Checks REA, its engines (Ghidra, Hopper, IDA) and agent registrations."},
    "providers": {"cmd": "providers", "target": False, "query": None, "provider": False,
                  "snapshot": False, "label": "List providers",
                  "help": "The analysis engines REA can use here and what each supports."},
    "capabilities": {"cmd": "capabilities", "target": False, "query": None, "provider": False,
                     "snapshot": False, "label": "List capabilities",
                     "help": "Everything this REA installation can do."},
}

# Uses the studio has for REA, shown on the page so the tool has a purpose, not just buttons.
USES = [
    ("A ComfyUI custom node pack or a model loader that misbehaves",
     "Analyse its folder to see what it imports, calls and writes before trusting it."),
    ("ComfyUI Desktop, Ollama or another Electron/native tool on the PC",
     "Analyse the app or inspect its .asar to learn how a feature works, with evidence, "
     "before building the same into the studio."),
    ("A codec, encoder or plugin whose behaviour is undocumented",
     "Search for a setting's name, then trace it to the code that uses it."),
    ("Readiness", "Doctor and providers show whether Ghidra, Hopper or IDA are set up for "
                  "native targets; JavaScript targets need no engine."),
]
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ADDRESS = re.compile(r"^(0x[0-9a-fA-F]+|[0-9]+)$")


class ReaStore:
    """Files of REA runs and snapshots, under ``<data_dir>/rea``."""

    def __init__(self, data_dir: Path) -> None:
        self.root = (Path(data_dir) / "rea").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, run_id: str) -> Path:
        if not _SAFE.match(run_id):
            raise ValueError(f"unsafe run id {run_id!r}")
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path_for(self, rel_path: str) -> Path:
        path = (self.root / rel_path).resolve()
        if self.root not in path.parents:
            raise ValueError(f"path escapes the REA folder: {rel_path!r}")
        return path

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    def snapshot_for(self, target: Path) -> Path:
        """One snapshot file per target path, so repeated queries reuse REA's cache."""
        folder = self.root / "snapshots"
        folder.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(str(target.resolve()).encode()).hexdigest()[:16]
        return folder / f"{target.name}_{digest}.json"


@dataclass
class Availability:
    ok: bool
    command: list[str]
    reason: str = ""
    version: str = ""


def command_for(settings: Settings) -> list[str]:
    return shlex.split(settings.rea.command or "rea")


def availability(settings: Settings, *, probe: bool = True) -> Availability:
    """Whether the worker can run REA here, and why not when it cannot."""
    cmd = command_for(settings)
    if not cmd:
        return Availability(False, cmd, "rea.command is empty")
    exe = shutil.which(cmd[0])
    if exe is None:
        hint = ("Install REA where the worker runs: Node.js 22+ and `npm install --global "
                "rea-agents` (or set rea.command to `npx -y rea-agents@latest`). In Docker "
                "the studio image has no Node.js; run the studio without Docker for REA, or "
                "point rea.command at a wrapper on the host.")
        return Availability(False, cmd, f"`{cmd[0]}` is not on the worker's PATH. {hint}")
    if not probe:
        return Availability(True, cmd)
    try:
        proc = subprocess.run([*cmd, "--version"], capture_output=True, text=True, timeout=60,
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Availability(False, cmd, f"`{cmd[0]} --version` failed: {exc}")
    text = (proc.stdout or proc.stderr).strip().splitlines()
    version = text[-1].strip() if text else ""
    if proc.returncode != 0:
        return Availability(False, cmd, f"`{cmd[0]} --version` exited {proc.returncode}: {version}")
    return Availability(True, cmd, version=version)


class ReaRequest(BaseModel):
    """One REA run as a person asks for it."""

    preset: Preset = "analyze"
    target: str = ""                     # a file or folder the worker can read
    query: str = Field("", max_length=2000)
    provider: str = Field("", max_length=64)
    extra_args: list[str] = Field(default_factory=list)  # passed through, e.g. ["--limit", "50"]
    project_id: str | None = None
    title: str = Field("", max_length=200)


def build_args(settings: Settings, store: ReaStore, request: ReaRequest) -> list[str]:
    """The argument list after ``rea``, always with ``--json``."""
    spec = PRESETS.get(request.preset)
    if spec is None:
        raise ValueError(f"unknown preset {request.preset!r}; known: {', '.join(PRESETS)}")
    args: list[str] = [spec["cmd"]]
    target: Path | None = None
    if spec["target"]:
        if not request.target.strip():
            raise ValueError("Give the file or folder to look at (a path the worker can read).")
        target = Path(request.target.strip()).expanduser()
        if not target.is_absolute():
            raise ValueError("Use an absolute path, so the worker and you mean the same file.")
        if not target.exists():
            raise ValueError(f"The worker cannot read {target}. Is it on this machine (in "
                             "Docker: under data/ or media/)?")
        args.append(str(target))
    elif request.target.strip():
        raise ValueError(f"{spec['label']} takes no target.")
    if spec["query"]:
        if not request.query.strip():
            raise ValueError(f"{spec['query']} is needed.")
        if "address (0x" in spec["query"].lower() and not _ADDRESS.match(request.query.strip()):
            raise ValueError("Give an address such as 0x1000.")
        args.append(request.query.strip())
    provider = request.provider.strip() or (settings.rea.provider if spec["provider"] else "")
    if provider and spec["provider"]:
        if not _SAFE.match(provider):
            raise ValueError("The provider id has letters, digits, dots, dashes or underscores.")
        args += ["--provider", provider]
    if spec["snapshot"] and settings.rea.snapshots and target is not None:
        args += ["--snapshot", str(store.snapshot_for(target))]
    for extra in request.extra_args:
        if not isinstance(extra, str) or extra.startswith("-") and extra.lstrip("-") == "":
            raise ValueError("extra arguments must be plain options and values")
        args.append(extra)
    args.append("--json")
    return args


def request_run(session: Session, settings: Settings, store: ReaStore, request: ReaRequest, *,
                actor: str = "api") -> ReaRun:
    """Validate and queue one ``rea`` job; the worker runs the command."""
    args = build_args(settings, store, request)
    avail = availability(settings, probe=False)
    if not avail.ok:
        raise ValueError(avail.reason)
    row = ReaRun(preset=request.preset, target=request.target.strip(), query=request.query.strip(),
                 args=args, status="queued", project_id=request.project_id,
                 title=request.title.strip())
    session.add(row)
    session.flush()
    job = enqueue(session, "rea", payload={"run_id": row.id, "actor": actor}, priority=95,
                  max_attempts=1)
    assert job is not None
    row.job_id = job.id
    session.flush()
    return row


def execute(settings: Settings, store: ReaStore, run: ReaRun) -> dict[str, Any]:
    """Run REA for one row, write its output next to the row, and return what to store."""
    cmd = [*command_for(settings), *run.args]
    folder = store.dir(run.id)
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                              timeout=settings.rea.timeout_s, cwd=str(folder))
    except subprocess.TimeoutExpired:
        return {"status": "failed", "exit_code": None, "took_s": settings.rea.timeout_s,
                "error": {"code": "timeout", "message": f"REA ran longer than "
                                                          f"{settings.rea.timeout_s:g} s"}}
    except OSError as exc:
        return {"status": "failed", "exit_code": None, "took_s": 0.0,
                "error": {"code": "not_runnable", "message": str(exc)}}
    took = round(time.monotonic() - started, 2)
    (folder / "command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    (folder / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    out_path = folder / "output.json"
    data: Any = None
    text = proc.stdout or ""
    try:
        data = json.loads(text) if text.strip() else None
    except ValueError:
        data = None
    if data is not None:
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        out_path = folder / "output.txt"
        out_path.write_text(text, encoding="utf-8")
    result: dict[str, Any] = {"exit_code": proc.returncode, "took_s": took,
                              "rel_path": store.rel(out_path), "summary": summarize(data)}
    if proc.returncode == 0:
        result["status"], result["error"] = "done", None
    else:
        message = (_error_message(data) or (proc.stderr or text).strip().splitlines()[-1:]
                   or ["REA exited with an error"])
        result["status"] = "failed"
        result["error"] = {"code": f"exit_{proc.returncode}",
                           "message": message if isinstance(message, str) else message[0]}
    return result


def _error_message(data: Any) -> str | None:
    """REA's error contract: ``error``, ``code``, ``message`` and a ``remediation.action``."""
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if isinstance(err, dict):
        err = err.get("message") or err.get("code")
    message = data.get("message") if isinstance(data.get("message"), str) else None
    text = message or (err if isinstance(err, str) else None)
    if not text:
        return None
    remedy = data.get("remediation")
    if isinstance(remedy, dict) and isinstance(remedy.get("action"), str):
        text += f" {remedy['action']}"
    return text


def summarize(data: Any) -> dict[str, Any]:
    """A short, honest digest of REA's JSON for the run list: what it looked at, which
    provider answered, how sure it is, how much it found and what stayed unresolved.
    Nothing is inferred beyond what the output states."""
    if not isinstance(data, dict):
        return {"kind": "text" if isinstance(data, str) else "none"}
    out: dict[str, Any] = {}
    subject = data.get("subject") if isinstance(data.get("subject"), dict) else {}
    provider = data.get("provider") if isinstance(data.get("provider"), dict) else {}
    if subject:
        out["subject"] = {k: subject.get(k) for k in ("name", "format", "architecture")
                          if subject.get(k) is not None}
    if provider:
        out["provider"] = provider.get("name") or provider.get("id")
    for key in ("operation", "confidence", "authority", "evidence_id"):
        if isinstance(data.get(key), str):
            out[key] = data[key]
    if isinstance(data.get("limitations"), list):
        out["limitations"] = len(data["limitations"])
    normalized = data.get("normalized_result")
    if isinstance(normalized, dict):
        stats = normalized.get("statistics")
        if isinstance(stats, dict):
            out["statistics"] = {k: v for k, v in stats.items() if isinstance(v, int | float)}
        semantic = normalized.get("semantic_graph")
        if isinstance(semantic, dict):
            out["graph"] = {k: len(semantic[k]) for k in ("nodes", "relations", "unknowns")
                            if isinstance(semantic.get(k), list)}
        if isinstance(normalized.get("integrity_contradictions"), list):
            out["integrity_contradictions"] = len(normalized["integrity_contradictions"])
        for key in ("functions", "strings", "matches", "results", "xrefs", "callers", "callees",
                    "instructions", "symbols"):
            if isinstance(normalized.get(key), list):
                out[key] = len(normalized[key])
    if "healthy" in data:  # doctor
        raw_checks = data.get("scope_checks")
        checks: list[dict[str, Any]] = [c for c in raw_checks if isinstance(c, dict)] \
            if isinstance(raw_checks, list) else []
        out["healthy"] = bool(data["healthy"])
        out["checks"] = {"ok": sum(1 for c in checks if c.get("ok")), "total": len(checks)}
        out["failing"] = [str(c.get("name")) for c in checks if not c.get("ok")][:8]
    if isinstance(data.get("capabilities"), list):  # capabilities / providers
        caps = [c for c in data["capabilities"] if isinstance(c, dict)]
        out["capabilities"] = {"available": sum(1 for c in caps if c.get("available")),
                               "total": len(caps)}
        candidates = data.get("analysis_provider_candidates")
        if isinstance(candidates, list):
            out["providers"] = {
                str((c.get("provider") or {}).get("id")): str((c.get("availability") or {})
                                                              .get("status"))
                for c in candidates if isinstance(c, dict)}
    if isinstance(data.get("error"), str | dict) or isinstance(data.get("code"), str):
        out["error"] = _error_message(data)
        if isinstance(data.get("code"), str):
            out["code"] = data["code"]
    if not out:
        out["keys"] = list(data.keys())[:12]
    return out


def list_runs(session: Session, *, limit: int = 60, offset: int = 0, preset: str | None = None,
              project_id: str | None = None) -> list[ReaRun]:
    stmt = (select(ReaRun).order_by(ReaRun.created_at.desc(), ReaRun.id.desc())
            .offset(offset).limit(limit))
    if preset:
        stmt = stmt.where(ReaRun.preset == preset)
    if project_id:
        stmt = stmt.where(ReaRun.project_id == project_id)
    return list(session.scalars(stmt))


def get_run(session: Session, run_id: str) -> ReaRun:
    run = session.get(ReaRun, run_id)
    if run is None:
        raise LookupError(f"REA run {run_id} not found")
    return run


def delete_run(session: Session, store: ReaStore, run: ReaRun) -> None:
    job = session.get(Job, run.job_id) if run.job_id else None
    if run.status in ("queued", "running") and job is not None and job.status in ("QUEUED", "RETRY_WAIT"):
        job.status = "CANCELLED"
        job.finished_at = utcnow()
    session.delete(run)
    session.flush()
    folder = store.root / run.id
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def output_of(store: ReaStore, run: ReaRun) -> Any:
    """The run's output: parsed JSON when it is JSON, else the text."""
    if not run.rel_path:
        return None
    path = store.path_for(run.rel_path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return text


def run_view(run: ReaRun) -> dict[str, Any]:
    return {"id": run.id, "preset": run.preset, "label": PRESETS.get(run.preset, {}).get("label", run.preset),
            "target": run.target, "query": run.query, "args": run.args, "status": run.status,
            "exit_code": run.exit_code, "project_id": run.project_id, "summary": run.summary,
            "error": run.error, "title": run.title, "took_s": run.took_s,
            "output_url": f"/rea/{run.id}/output" if run.rel_path else None,
            "created_at": run.created_at.isoformat() if isinstance(run.created_at, datetime) else None,
            "finished_at": run.finished_at.isoformat() if isinstance(run.finished_at, datetime) else None}
