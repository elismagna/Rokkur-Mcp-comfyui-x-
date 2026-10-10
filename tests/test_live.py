"""The live view: what runs, what each stage produced, decisions with evidence, events."""

from __future__ import annotations

from rokkur_studio.domain.rights import RightsCategory
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.services import live
from rokkur_studio.services.projects import get_project
from tests.test_dashboard import client_for
from tests.test_pipeline import create, run, status


def snap(ctx, pid: str) -> dict:
    with ctx.db.session() as s:
        return live.snapshot(s, ctx, get_project(s, pid))


def test_the_snapshot_follows_a_video_from_creation_to_the_final_cut(ctx, sample_video):
    pid = create(ctx, sample_video)
    s0 = snap(ctx, pid)
    assert [n["id"] for n in s0["nodes"]][:3] == ["rights", "ingest", "analyse"]
    states = {n["id"]: n["state"] for n in s0["nodes"]}
    assert states["rights"] == "current" and states["ingest"] == "todo" and s0["shots"] == []
    assert s0["decisions"] == [] and s0["now"]["queued"][0]["kind"] == "rights_check"
    assert s0["working"] is True and s0["now"]["job"] is None
    assert any(e["type"] == "PROJECT_CREATED" for e in s0["events"])
    Worker(ctx, worker_id="t").drain(max_jobs=3)  # rights, ingest, analyse
    s1 = snap(ctx, pid)
    states = {n["id"]: n["state"] for n in s1["nodes"]}
    assert states["rights"] == "done" and states["analyse"] == "current" and states["brief"] == "todo"
    assert s1["now"]["queued"][0]["kind"] == "creative_plan"  # analysed; the brief is next
    assert s1["working"] is True and len(s1["shots"]) >= 2
    assert all(sh["state"] == "planned" and sh["intent"].endswith("motion") for sh in s1["shots"])
    trace = {st["id"]: st for st in s1["stages"]}
    assert "rights gate" in trace["rights"]["by"] and "user owned" in trace["rights"]["lines"][0]
    assert "fps" in trace["ingest"]["lines"][0] and "scene cuts" in trace["analyse"]["lines"][0]
    assert trace["brief"]["lines"] == [] and trace["brief"]["state"] == "todo"
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    s2 = snap(ctx, pid)
    states = {n["id"]: n["state"] for n in s2["nodes"]}
    assert states["ready"] == "done" and states["published"] == "todo" and states["repair"] == "skipped"
    assert all(sh["state"] == "rendered" and sh["render_url"] and sh["qc"]["decision"] == "PASS"
               and sh["keyframe_url"] for sh in s2["shots"])
    trace = {st["id"]: st for st in s2["stages"]}
    assert "no model" in trace["brief"]["by"] and "shots planned" in trace["brief"]["lines"][-1]
    assert trace["render"]["lines"][0].startswith(f"{len(s2['shots'])} of {len(s2['shots'])} shots rendered")
    assert "of 10" in trace["quality"]["lines"][0] and trace["quality"]["by"].startswith("measured")
    assert "final/final.mp4" in trace["edit"]["lines"][0] and trace["edit"]["by"] == "FFmpeg"
    assert trace["ready"]["lines"][0]  # the drafted title
    texts = [e["text"] for e in s2["events"]]
    assert any(t.startswith("shot 001 rendered in") for t in texts)
    assert any(t.startswith("quality ") and "pass" in t for t in texts)
    assert any("final video encoded" in t for t in texts)
    assert s2["events"][0]["id"] > s2["events"][-1]["id"]  # newest first
    assert s2["usage"]["renders"] == len(s2["shots"]) and s2["now"]["job"] is None


def test_decisions_carry_their_evidence_and_their_actions_work(ctx, sample_video):
    # a rights question: the person sees the frames and decides
    pid = create(ctx, sample_video, category=RightsCategory.UNKNOWN, evidence="")
    run(ctx)
    assert status(ctx, pid) == "RIGHTS_PENDING"
    s = snap(ctx, pid)
    assert {n["id"]: n["state"] for n in s["nodes"]}["rights"] == "waiting"
    [decision] = s["decisions"]
    assert decision["kind"] == "rights" and decision["actions"][0]["url"].endswith("/approve-rights")
    c = client_for(ctx)
    page = c.get(f"/ui/projects/{pid}/live").text
    assert "Waiting for you" in page and "May the studio use this footage?" in page
    assert "Approve rights" in page and 'id="live-snapshot"' in page
    # the repair limit: failing renders next to their original frames, with the three ways out
    ctx.settings.render.max_retries = 1
    pid2 = create(ctx, sample_video, test_faults={"shot_002": {"kind": "black", "attempts": [1, 2, 3]}})
    run(ctx)
    assert status(ctx, pid2) == "FAILED"
    s = snap(ctx, pid2)
    [decision] = s["decisions"]
    assert decision["kind"] == "repair_limit" and "shot 002" in decision["why"]
    kinds = [(ev["kind"], ev["caption"]) for ev in decision["evidence"]]
    assert ("image", "Shot 002 · original") in kinds
    assert any(k == "video" and "render" in cap for k, cap in kinds)
    assert [a["label"] for a in decision["actions"]][:3] == ["Try 1 more repairs", "Check quality again",
                                                              "Keep these renders"]
    shot2 = next(sh for sh in s["shots"] if sh["id"] == "shot_002")
    assert shot2["state"] == "needs repair" and shot2["attempts"] == 2 and shot2["qc"]["issues"]
    assert {n["id"]: n["state"] for n in s["nodes"]}["quality"] == "failed"
    page = c.get(f"/ui/projects/{pid2}/live").text
    assert "Stopped after 1 repair round" in page and "Keep these renders" in page
    assert c.get(f"/ui/projects/{pid2}/live.json").json()["decisions"][0]["kind"] == "repair_limit"
    api = c.get(f"/projects/{pid2}/live?events=5").json()
    assert len(api["events"]) == 5 and api["project"]["status"] == "FAILED"
    assert c.get("/projects/nope/live").status_code == 404
    r = c.post(f"/ui/projects/{pid2}/keep-renders", follow_redirects=False)
    assert r.status_code == 303
    run(ctx)
    assert status(ctx, pid2) == "READY_TO_PUBLISH"
    s = snap(ctx, pid2)
    assert s["decisions"] == [] and any("kept by you" in line for line in
                                        next(st for st in s["stages"] if st["id"] == "quality")["lines"])
    assert "Watch it work" in c.get(f"/ui/projects/{pid2}").text


def test_a_stopped_video_offers_resume_and_the_queue_is_shown(ctx, sample_video):
    ctx.settings.render.renderer = "comfyui"
    pid = create(ctx, sample_video)
    original = ctx.settings.profiles["PREVIEW"].workflow
    ctx.settings.profiles["PREVIEW"].workflow = "removed_after_creation"
    run(ctx)
    ctx.settings.profiles["PREVIEW"].workflow = original
    assert status(ctx, pid) == "FAILED"
    s = snap(ctx, pid)
    [decision] = s["decisions"]
    assert decision["kind"] == "failed" and decision["actions"][0]["label"] == "Resume"
    assert "workflow" in decision["why"].lower()
    assert {n["id"]: n["state"] for n in s["nodes"]}["workflow"] == "failed"
    pid2 = create(ctx, sample_video)
    s = snap(ctx, pid2)
    assert s["now"]["queued"] and s["now"]["queued"][0]["label"] == "Checking rights"
    assert s["working"] is True
