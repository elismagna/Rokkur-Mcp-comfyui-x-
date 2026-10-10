# ruff: noqa: E501
"""REA as a studio tool: a fake ``rea`` that answers like the real 6.3.0 CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rokkur_studio.db.models import ReaRun
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.services import rea as svc
from rokkur_studio.services.rea import ReaRequest, ReaStore, build_args, summarize
from tests.test_dashboard import client_for

FAKE = r'''
import json, sys, os
args = sys.argv[1:]
if args == ["--version"]:
    print("rea@6.3.0"); sys.exit(0)
cmd = args[0]
def out(data, code=0):
    print(json.dumps(data)); sys.exit(code)
if cmd == "doctor":
    out({"healthy": False, "scope_checks": [{"name": "node", "ok": True}, {"name": "ghidra", "ok": False,
         "classification": "missing_analysis_engine"}]}, 1)
if cmd in ("capabilities", "providers"):
    out({"open": False, "capabilities": [{"operation": "analyze_javascript_application", "available": True},
         {"operation": "capture_native_ui_scenario", "available": False}],
         "analysis_provider_candidates": [{"provider": {"id": "ghidra"}, "availability": {"status": "unavailable"}}]})
target = args[1] if len(args) > 1 else ""
if cmd == "analyze" and os.path.isdir(target):
    sys.stderr.write('{"rea_progress":{"phase":"x","completed":1,"total":1}}\n')
    out({"evidence_id": "ev_1", "subject": {"name": os.path.basename(target), "format": "directory", "architecture": None},
         "provider": {"id": "rea-javascript-application", "name": "REA JavaScript application analyzer"},
         "operation": "analyze_javascript_application", "confidence": "high", "authority": "static",
         "limitations": ["a", "b"], "normalized_result": {"statistics": {"relevant_files": 2, "modules": 1, "findings": 3},
         "semantic_graph": {"nodes": [1, 2, 3], "relations": [1], "unknowns": [1, 2]}, "integrity_contradictions": []},
         "snapshot": [a for a in args if a == "--snapshot"]})
if cmd in ("inspect-artifact", "search", "function", "decompile", "xrefs", "trace", "instructions") and os.path.isdir(target):
    out({"error": "Analysis failed", "code": "target_unavailable", "category": "unavailable",
         "message": "open_binary accepts files and macOS app bundles. This target is a directory.",
         "remediation": {"action": "For a JavaScript/Electron application directory run `rea analyze <directory>`."}}, 1)
if cmd == "search":
    out({"subject": {"name": os.path.basename(target), "format": "elf"}, "provider": {"id": "ghidra", "name": "Ghidra"},
         "normalized_result": {"matches": [{"text": args[2]}, {"text": args[2] + "2"}]}})
if cmd == "analyze":
    out({"subject": {"name": os.path.basename(target), "format": "elf", "architecture": "x86_64"},
         "provider": {"id": args[args.index("--provider") + 1] if "--provider" in args else "auto", "name": "P"},
         "normalized_result": {"functions": [1, 2, 3, 4]}})
print("not json at all"); sys.exit(1)
'''


@pytest.fixture
def fake_rea(ctx, tmp_path) -> Path:
    script = tmp_path / "fake_rea.py"
    script.write_text(FAKE)
    ctx.settings.rea.command = f"{sys.executable} {script}"
    return script


def store_for(ctx) -> ReaStore:
    return ReaStore(ctx.settings.studio.data_dir)


def request(ctx, **fields) -> ReaRun:
    with ctx.db.transaction() as s:
        row = svc.request_run(s, ctx.settings, store_for(ctx), ReaRequest(**fields))
        s.refresh(row)
        return row


def run_of(ctx, run_id: str) -> ReaRun:
    with ctx.db.session() as s:
        return s.get(ReaRun, run_id)


def test_availability_says_how_to_install_rea_and_reports_its_version(ctx, fake_rea):
    ctx.settings.rea.command = "rea-not-here"
    avail = svc.availability(ctx.settings)
    assert not avail.ok and "npm install --global rea-agents" in avail.reason
    with pytest.raises(ValueError, match="not on the worker"):
        request(ctx, preset="doctor")
    ctx.settings.rea.command = f"{sys.executable} {fake_rea}"
    avail = svc.availability(ctx.settings)
    assert avail.ok and avail.version == "rea@6.3.0"


def test_arguments_follow_the_rea_cli_and_are_checked_first(ctx, fake_rea, tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    store = store_for(ctx)
    args = build_args(ctx.settings, store, ReaRequest(preset="analyze", target=str(app)))
    assert args[:2] == ["analyze", str(app)] and args[-1] == "--json"
    assert "--snapshot" in args and args[args.index("--snapshot") + 1].startswith(str(store.root / "snapshots"))
    ctx.settings.rea.provider = "ghidra"
    args = build_args(ctx.settings, store, ReaRequest(preset="search", target=str(app), query="hello"))
    assert args[:3] == ["search", str(app), "hello"] and args[args.index("--provider") + 1] == "ghidra"
    assert build_args(ctx.settings, store, ReaRequest(preset="providers")) == ["providers", "--json"]
    doctor = build_args(ctx.settings, store, ReaRequest(preset="doctor", provider="hopper"))
    assert doctor == ["doctor", "--provider", "hopper", "--json"]
    with pytest.raises(ValueError, match="Give the file or folder"):
        build_args(ctx.settings, store, ReaRequest(preset="analyze"))
    with pytest.raises(ValueError, match="absolute path"):
        build_args(ctx.settings, store, ReaRequest(preset="analyze", target="app"))
    with pytest.raises(ValueError, match="cannot read"):
        build_args(ctx.settings, store, ReaRequest(preset="analyze", target=str(tmp_path / "nope")))
    with pytest.raises(ValueError, match="is needed"):
        build_args(ctx.settings, store, ReaRequest(preset="search", target=str(app)))
    with pytest.raises(ValueError, match="address such as"):
        build_args(ctx.settings, store, ReaRequest(preset="decompile", target=str(app), query="main"))
    with pytest.raises(ValueError, match="takes no target"):
        build_args(ctx.settings, store, ReaRequest(preset="doctor", target=str(app)))
    with pytest.raises(ValueError, match="letters, digits"):
        build_args(ctx.settings, store, ReaRequest(preset="analyze", target=str(app), provider="a b"))


def test_runs_keep_the_real_output_and_summarise_it_honestly(ctx, fake_rea, tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.js").write_text("console.log(1)")
    analysed = request(ctx, preset="analyze", target=str(app), title="The app")
    assert analysed.status == "queued" and analysed.job_id
    assert Worker(ctx, worker_id="t").drain() == 1
    done = run_of(ctx, analysed.id)
    assert done.status == "done" and done.exit_code == 0 and done.rel_path.endswith("output.json")
    assert done.summary["subject"] == {"name": "app", "format": "directory"}
    assert done.summary["provider"] == "REA JavaScript application analyzer"
    assert done.summary["confidence"] == "high" and done.summary["limitations"] == 2
    assert done.summary["statistics"]["findings"] == 3 and done.summary["graph"] == {"nodes": 3, "relations": 1, "unknowns": 2}
    output = svc.output_of(store_for(ctx), done)
    assert output["evidence_id"] == "ev_1" and output["snapshot"] == ["--snapshot"]
    folder = store_for(ctx).root / done.id
    assert (folder / "command.txt").read_text().startswith(sys.executable.split("/")[-1][:3] or "p")
    assert "rea_progress" in (folder / "stderr.txt").read_text()
    # REA's error contract becomes the run's error, with its remediation
    wrong = request(ctx, preset="inspect", target=str(app))
    Worker(ctx, worker_id="t").drain()
    wrong = run_of(ctx, wrong.id)
    assert wrong.status == "failed" and wrong.exit_code == 1
    assert "This target is a directory" in wrong.error["message"] and "rea analyze" in wrong.error["message"]
    assert wrong.summary["code"] == "target_unavailable"
    # doctor exits 1 when an engine is missing: that is a result, kept with its checks
    doc = request(ctx, preset="doctor")
    Worker(ctx, worker_id="t").drain()
    doc = run_of(ctx, doc.id)
    assert doc.status == "failed" and doc.summary == {"healthy": False, "checks": {"ok": 1, "total": 2},
                                                      "failing": ["ghidra"]}
    caps = request(ctx, preset="capabilities")
    Worker(ctx, worker_id="t").drain()
    caps = run_of(ctx, caps.id)
    assert caps.summary["capabilities"] == {"available": 1, "total": 2}
    assert caps.summary["providers"] == {"ghidra": "unavailable"}
    binary = tmp_path / "prog"
    binary.write_bytes(b"\x7fELF")
    found = request(ctx, preset="search", target=str(binary), query="hello", provider="ghidra")
    Worker(ctx, worker_id="t").drain()
    assert run_of(ctx, found.id).summary["matches"] == 2
    assert summarize("plain") == {"kind": "text"} and summarize({"odd": 1}) == {"keys": ["odd"]}
    with ctx.db.transaction() as s:
        svc.delete_run(s, store_for(ctx), svc.get_run(s, done.id))
    assert run_of(ctx, done.id) is None and not folder.exists()
    with ctx.db.session() as s:
        assert [r.preset for r in svc.list_runs(s)] == ["search", "capabilities", "doctor", "inspect"]


def test_rea_page_api_and_cli(ctx, fake_rea, tmp_path, monkeypatch, capsys):
    from rokkur_studio import cli
    from rokkur_studio.pipeline import context as context_mod

    c = client_for(ctx)
    ctx.settings.rea.command = "rea-not-here"
    page = c.get("/ui/rea").text
    assert "REA is not available to the worker" in page and "npm install --global rea-agents" in page
    ctx.settings.rea.command = f"{sys.executable} {fake_rea}"
    page = c.get("/ui/rea").text
    assert "rea@6.3.0" in page and "No runs yet" in page and "What the studio uses it for" in page
    app = tmp_path / "app"
    app.mkdir()
    r = c.post("/ui/rea", data={"preset": "analyze", "target": str(app), "extra_args": "--limit 5",
                                "title": "Electron app"}, follow_redirects=False)
    assert r.status_code == 303 and "run=rea_" in r.headers["location"]
    run_id = r.headers["location"].split("run=")[1].split("&")[0].split("#")[0]
    assert "Waiting for a worker" in c.get(f"/ui/rea?run={run_id}").text
    Worker(ctx, worker_id="t").drain()
    page = c.get(f"/ui/rea?run={run_id}").text
    assert "Electron app" in page and "REA JavaScript application analyzer" in page and "ev_1" in page
    assert "--limit 5" in page and "findings 3" in page
    r = c.post("/ui/rea", data={"preset": "search", "target": str(app)}, follow_redirects=False)
    assert "err=Text%20to%20search" in r.headers["location"]
    # the API
    status = c.get("/rea/status").json()
    assert status["available"] and status["version"] == "rea@6.3.0" and "analyze" in status["presets"]
    r = c.post("/rea", json={"preset": "doctor"})
    assert r.status_code == 202 and r.json()["status"] == "queued"
    assert c.post("/rea", json={"preset": "function", "target": str(app)}).status_code == 422
    Worker(ctx, worker_id="t").drain()
    listed = c.get("/rea?limit=5").json()
    assert listed[0]["preset"] == "doctor" and listed[0]["status"] == "failed"
    out = c.get(f"/rea/{run_id}/output")
    assert out.status_code == 200 and json.loads(out.content)["evidence_id"] == "ev_1"
    assert c.delete(f"/rea/{listed[0]['id']}").status_code == 204
    assert c.get(f"/rea/{listed[0]['id']}").status_code == 404
    # the CLI
    monkeypatch.setattr(cli, "_settings", lambda args: ctx.settings)
    monkeypatch.setattr(context_mod, "build_context", lambda settings, db=None: ctx)
    assert cli.main(["rea-run", "capabilities", "--wait"]) == 0
    text = capsys.readouterr().out
    assert "done" in text and "capabilities: {'available': 1, 'total': 2}" in text
    assert cli.main(["rea-list"]) == 0 and "capabilities" in capsys.readouterr().out
    assert cli.main(["rea-run", "analyze"]) == 1
