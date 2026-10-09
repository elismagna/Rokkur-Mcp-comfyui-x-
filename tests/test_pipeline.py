"""End-to-end pipeline tests: real FFmpeg, real Postgres, in-process worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from rokkur_studio.api.schemas import CreativeIn, ProjectCreate, RightsIn, SourceIn
from rokkur_studio.comfyui.client import ComfyClient
from rokkur_studio.db.models import ApprovalRequest, Event, GpuLease, Job, Render
from rokkur_studio.domain.rights import RightsCategory
from rokkur_studio.domain.states import InvalidTransition, ProjectStatus
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.pipeline.renderers import FFmpegPreviewRenderer, RenderOOM, RenderUnavailable
from rokkur_studio.services import commands, publishing
from rokkur_studio.services.projects import (
    at_budget_limit,
    at_repair_limit,
    failure_reason,
    get_project,
    latest_document,
    transition,
)
from tests.fakes import FakeComfyUI


def create(ctx, video: Path, *, category=RightsCategory.USER_OWNED, evidence="mine",
           autostart=True, **creative: Any) -> str:
    with ctx.db.transaction() as s:
        p = commands.create_project(s, ProjectCreate(
            name="test", source=SourceIn(local_path=str(video)),
            rights=RightsIn(category=category, permission_evidence=evidence),
            creative=CreativeIn(theme="retro clay sci-fi", **creative), autostart=autostart),
            ctx.settings)
        return p.id


def run(ctx, rounds: int = 50) -> None:
    worker = Worker(ctx, worker_id="test")
    for _ in range(rounds):
        if not worker.run_once():
            return


def status(ctx, pid: str) -> str:
    with ctx.db.session() as s:
        return get_project(s, pid).status


def events(ctx, pid: str) -> list[str]:
    with ctx.db.session() as s:
        return [e.type for e in s.scalars(select(Event).where(Event.project_id == pid)
                                          .order_by(Event.id))]


def test_end_to_end_fixture_to_dry_run_publish(ctx, sample_video):
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    final = ctx.store.path_for(f"{pid}/final/final.mp4")
    info = ctx.ffmpeg.probe(final)
    assert (info.width, info.height, info.has_audio) == (1080, 1920, True)
    for area in ("source", "analysis", "manifests", "renders", "qc", "final", "thumbnails"):
        assert any(ctx.store.project_dir(pid, area).iterdir()), area
    with ctx.db.transaction() as s:
        project = get_project(s, pid)
        assert latest_document(s, pid, "qc_report").data["decision"] == "PASS"
        pub = publishing.dry_run(s, project)
        body = pub.request["body"]
        assert body["status"]["privacyStatus"] == "private"
        assert body["status"]["containsSyntheticMedia"] is True
        assert body["snippet"]["title"].endswith("#shorts")
    ev = events(ctx, pid)
    for expected in ("PROJECT_CREATED", "RIGHTS_APPROVED", "ANALYSIS_STARTED", "RENDER_SUBMITTED",
                     "RENDER_COMPLETED", "QC_PASSED", "FINAL_ENCODED", "PUBLISH_DRY_RUN"):
        assert expected in ev
    with ctx.db.session() as s:
        jobs = s.scalars(select(Job).where(Job.project_id == pid)).all()
        assert all(j.status == "SUCCEEDED" and j.duration_s is not None for j in jobs)


def test_end_to_end_mix_keeps_source_audio_under_optional_bed(ctx, sample_video):
    upload_dir = ctx.settings.studio.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    bed = upload_dir / "test-bed.wav"
    ctx.ffmpeg._run(ctx.ffmpeg._ff("-f", "lavfi", "-i", "sine=frequency=660:duration=1",
                                   "-c:a", "pcm_s16le", str(bed)))
    pid = create(ctx, sample_video, audio_bed_path=str(bed), audio_bed_gain=0.2,
                 keep_source_audio=True, audio_bed_rights_confirmed=True,
                 audio_bed_rights_evidence="Synthetic test tone")
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    info = ctx.ffmpeg.probe(ctx.store.path_for(f"{pid}/final/final.mp4"))
    assert info.has_audio and abs(info.duration - 4) < 0.1
    assert any("amix=inputs=2" in " ".join(item["cmd"]) for item in ctx.ffmpeg.history)


def test_bad_shot_is_repaired_without_rerendering_good_shots(ctx, sample_video):
    pid = create(ctx, sample_video,
                 test_faults={"shot_002": {"kind": "black", "attempts": [1]}})
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.session() as s:
        rows = s.scalars(select(Render).where(Render.project_id == pid)
                         .order_by(Render.shot_id, Render.attempt)).all()
        assert [(r.shot_id, r.attempt, r.status) for r in rows] == [
            ("shot_001", 1, "succeeded"), ("shot_002", 1, "superseded"),
            ("shot_002", 2, "succeeded")]
        plan = latest_document(s, pid, "repair_plan").data
        assert [a["shot_id"] for a in plan["actions"]] == ["shot_002"]
    repair_events = [e for e in events(ctx, pid)
                     if e in ("QC_FAILED", "REPAIR_REQUESTED", "QC_PASSED")]
    assert repair_events == ["QC_FAILED", "REPAIR_REQUESTED", "QC_PASSED"]


def test_repair_budget_stops_infinite_loops(ctx, sample_video):
    ctx.settings.render.max_retries = 1
    pid = create(ctx, sample_video,
                 test_faults={"shot_002": {"kind": "black", "attempts": [1, 2, 3, 4]}})
    run(ctx)
    assert status(ctx, pid) == "FAILED"
    assert "REPAIR_BUDGET_EXHAUSTED" in events(ctx, pid)
    with ctx.db.session() as s:
        assert s.scalar(select(ApprovalRequest.kind).where(ApprovalRequest.project_id == pid)) \
            == "repair_budget"


def test_repair_limit_is_extended_by_approval_or_overridden_by_keeping(ctx, sample_video):
    ctx.settings.render.max_retries = 1
    pid = create(ctx, sample_video,
                 test_faults={"shot_002": {"kind": "black", "attempts": [1, 2, 3]}})
    run(ctx)
    assert status(ctx, pid) == "FAILED"
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        assert at_repair_limit(project)
        assert failure_reason(s, project) == "repair budget exhausted"
        with pytest.raises(InvalidTransition, match="repair limit"):
            commands.resume_project(s, project, ctx.settings)
        with pytest.raises(ValueError, match="between 1 and 10"):
            commands.repair_more(s, project, ctx.settings, actor="elis", rounds=11)
        commands.recheck_quality(s, project, ctx.settings, actor="elis")
        assert project.status == "QUALITY_CHECK"
    run(ctx)  # QC scores the same renders again: still black, so it stops at the limit again
    assert status(ctx, pid) == "FAILED"
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        assert s.scalar(select(func.count(Render.id)).where(Render.project_id == pid,
                                                            Render.shot_id == "shot_002")) == 2
        req = s.scalars(select(ApprovalRequest).where(ApprovalRequest.project_id == pid,
                                                      ApprovalRequest.status == "pending")).one()
        commands.decide_approval(s, req, ctx.settings, approve=True, decided_by="elis",
                                 note=None)
        assert req.status == "approved" and req.note == "1 more repair rounds"
        assert project.status == "QUALITY_FAILED"
    run(ctx)  # round 2 renders attempt 3, which is black again
    assert status(ctx, pid) == "FAILED"
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        assert project.repair_rounds == 2 and at_repair_limit(project)
        commands.keep_renders(s, project, ctx.settings, actor="elis", note="looks fine")
        assert project.status == "QUALITY_PASSED"
        qc = latest_document(s, pid, "qc_report").data
        assert qc["decision"] == "PASS"
        assert qc["override"]["by"] == "elis" and qc["override"]["failed_shots"] == ["shot_002"]
        assert all(r.status == "approved" for r in s.scalars(
            select(ApprovalRequest).where(ApprovalRequest.project_id == pid)))
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    ev = events(ctx, pid)
    assert ev.count("REPAIR_BUDGET_EXHAUSTED") == 3
    assert "REPAIR_BUDGET_EXTENDED" in ev and "QC_OVERRIDDEN" in ev
    with ctx.db.transaction() as s, pytest.raises(InvalidTransition, match="repair limit"):
        commands.keep_renders(s, get_project(s, pid, for_update=True), ctx.settings,
                              actor="elis")


def test_render_budget_stop_is_lifted_only_by_allowing_more_renders(ctx, sample_video):
    ctx.settings.render.max_renders_per_project = 1
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "FAILED"
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        assert at_budget_limit(s, project, ctx.settings) and not at_repair_limit(project)
        assert failure_reason(s, project) == "render budget exhausted (1/1)"
        with pytest.raises(InvalidTransition, match="render budget"):
            commands.resume_project(s, project, ctx.settings)
        with pytest.raises(ValueError, match="between 1 and 1"):
            commands.allow_more_renders(s, project, ctx.settings, actor="elis", renders=2)
        commands.allow_more_renders(s, project, ctx.settings, actor="elis")
        assert project.status == "WORKFLOW_READY"
    run(ctx)  # one more render covers the last shot
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.session() as s:
        ext = s.scalars(select(Event).where(Event.project_id == pid,
                                            Event.type == "BUDGET_EXTENDED")).one()
        assert ext.data == {"renders": 1, "gpu_minutes": 120.0, "cloud_minutes": 60.0}
        assert ext.actor == "elis"
        assert s.scalar(select(func.count(Render.id)).where(Render.project_id == pid)) == 2
    with ctx.db.transaction() as s, pytest.raises(InvalidTransition, match="not stopped"):
        commands.allow_more_renders(s, get_project(s, pid, for_update=True), ctx.settings,
                                    actor="elis")


def test_render_budget_stop_resumes_after_the_configured_budget_grows(ctx, sample_video):
    ctx.settings.render.max_renders_per_project = 1
    pid = create(ctx, sample_video)
    run(ctx)
    ctx.settings.render.max_renders_per_project = 40
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        assert project.status == "FAILED" and not at_budget_limit(s, project, ctx.settings)
        commands.resume_project(s, project, ctx.settings)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"


def test_empty_repair_plan_stops_where_renders_can_be_kept(ctx, sample_video, monkeypatch):
    from rokkur_studio.agents.roles import RepairPlanner
    from rokkur_studio.agents.schemas import RepairPlan

    monkeypatch.setattr(RepairPlanner, "plan", lambda self, report, round_, current, **kw:
                        RepairPlan(round=round_, actions=[]))
    pid = create(ctx, sample_video, test_faults={"shot_002": {"kind": "black"}})
    run(ctx)
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        assert at_repair_limit(project)
        assert failure_reason(s, project) == "this workflow has no automatic repair controls"
        commands.keep_renders(s, project, ctx.settings, actor="elis")
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"


def test_workflow_missing_at_compile_time_fails_the_project(ctx, sample_video):
    ctx.settings.render.renderer = "comfyui"
    pid = create(ctx, sample_video)  # the workflow is there when the project is created
    ctx.settings.profiles["PREVIEW"].workflow = "removed_after_creation"
    run(ctx)
    assert status(ctx, pid) == "FAILED"
    with ctx.db.session() as s:
        job = s.scalars(select(Job).where(Job.kind == "compile_workflow")).one()
        assert job.error["code"] == "template_error"


def test_failure_message_comes_from_the_failed_transition(ctx, sample_video):
    pid = create(ctx, sample_video, autostart=False)
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        transition(s, project, ProjectStatus.FAILED, actor="t", reason="the real cause")
        assert failure_reason(s, project) == "the real cause"
        assert not at_repair_limit(project) and not at_budget_limit(s, project, ctx.settings)


def test_unknown_rights_block_until_a_human_approves(ctx, sample_video):
    pid = create(ctx, sample_video, category=RightsCategory.UNKNOWN, evidence=None)
    run(ctx)
    assert status(ctx, pid) == "RIGHTS_PENDING"
    with ctx.db.transaction() as s:
        req = s.scalars(select(ApprovalRequest).where(ApprovalRequest.project_id == pid)).one()
        assert req.kind == "rights_ambiguity"
        commands.decide_approval(s, req, ctx.settings, approve=True, decided_by="elis",
                                 note="I filmed this")
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"


def test_reference_only_sources_are_rejected(ctx, sample_video):
    pid = create(ctx, sample_video, category=RightsCategory.REFERENCE_ONLY)
    run(ctx)
    assert status(ctx, pid) == "RIGHTS_REJECTED"
    assert not ctx.store.project_dir(pid, "source").exists() or not any(
        ctx.store.project_dir(pid, "source").iterdir())


def test_youtube_source_without_file_is_never_downloaded(ctx):
    with ctx.db.transaction() as s:
        p = commands.create_project(s, ProjectCreate(
            name="yt", source=SourceIn(platform="youtube", video_id="dQw4w9WgXcQ"),
            rights=RightsIn(category=RightsCategory.CREATIVE_COMMONS),
            creative=CreativeIn(theme="x"), autostart=True), ctx.settings)
        pid = p.id
    run(ctx)
    assert status(ctx, pid) == "FAILED"
    with ctx.db.session() as s:
        job = s.scalars(select(Job).where(Job.kind == "ingest")).one()
        assert "never downloads" in job.error["message"] and job.retry_count == 1


class FlakyRenderer(FFmpegPreviewRenderer):
    def __init__(self, ffmpeg, failures: list[type[Exception]]) -> None:
        super().__init__(ffmpeg)
        self.failures = failures
        self.calls: list[dict[str, Any]] = []

    def render_shot(self, **kw):
        self.calls.append(dict(kw["params"]))
        if self.failures:
            raise self.failures.pop(0)("simulated")
        return super().render_shot(**kw)


def test_transient_renderer_outage_is_retried(ctx, sample_video):
    ctx.extras["renderer"] = FlakyRenderer(ctx.ffmpeg, [RenderUnavailable])
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.session() as s:
        render_job = s.scalars(select(Job).where(Job.kind == "render")).one()
        assert render_job.retry_count == 1 and render_job.status == "SUCCEEDED"


def test_oom_degrades_instead_of_retrying_identical_work(ctx, sample_video):
    renderer = FlakyRenderer(ctx.ffmpeg, [RenderOOM, RenderOOM])
    renderer.name = "fake_gpu"
    ctx.extras["renderer"] = renderer
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    first, _, third = renderer.calls[:3]
    assert third["FPS"] < first["FPS"]
    with ctx.db.session() as s:
        steps = [e.data["recovery_step"] for e in s.scalars(
            select(Event).where(Event.type == "GPU_OOM").order_by(Event.id))]
    assert steps == ["clear_cache", "reduce_frames"]  # each retry follows a new recovery step
    with ctx.db.session() as s:
        assert s.scalars(select(GpuLease).where(GpuLease.released_at.is_(None))).all() == []


def test_oom_ladder_exhaustion_escalates(ctx, sample_video):
    ctx.extras["renderer"] = FlakyRenderer(ctx.ffmpeg, [RenderOOM] * 10)
    ctx.extras["renderer"].name = "fake_gpu"
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "FAILED"
    with ctx.db.session() as s:
        job = s.scalars(select(Job).where(Job.kind == "render")).one()
        assert job.error["code"] == "oom_unrecoverable" and job.retry_count == 1


def test_failed_project_resumes_where_it_stopped(ctx, sample_video):
    ctx.extras["renderer"] = FlakyRenderer(ctx.ffmpeg, [RenderUnavailable] * 3)
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "FAILED"
    with ctx.db.transaction() as s:
        commands.resume_project(s, get_project(s, pid, for_update=True), ctx.settings)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    assert "PROJECT_RESUMED" in events(ctx, pid)


def test_cancel_stops_queued_work(ctx, sample_video):
    pid = create(ctx, sample_video)
    with ctx.db.transaction() as s:
        commands.cancel(s, get_project(s, pid, for_update=True))
    run(ctx)
    assert status(ctx, pid) == "CANCELLED"
    with ctx.db.session() as s:
        assert {j.status for j in s.scalars(select(Job))} == {"CANCELLED"}


def test_autonomy_level_zero_waits_for_manual_advance(ctx, sample_video):
    ctx.settings.studio.autonomy_level = 0
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "RIGHTS_OK"  # start ran rights_check, then nothing automatic
    from rokkur_studio.pipeline.driver import advance

    with ctx.db.transaction() as s:
        advance(s, get_project(s, pid), ctx.settings, manual=True)
    run(ctx)
    assert status(ctx, pid) == "DOWNLOADED_OR_INGESTED"


def test_autonomy_level_one_stops_after_creative_brief(ctx, sample_video):
    ctx.settings.studio.autonomy_level = 1
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "CREATIVE_READY"


def test_comfyui_render_path_with_fake_server(ctx, sample_video):
    fake = FakeComfyUI()
    ctx.settings.render.renderer = "comfyui"
    ctx.settings.comfyui.poll_interval_s = 0
    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=fake.transport())
    hooks: list[str] = []
    ctx.gpu.before_heavy = [lambda h: hooks.append("unload_ollama")]
    ctx.gpu.after_heavy = [lambda h: hooks.append("free_comfyui")]
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    assert len(fake.prompts) == 2  # one ComfyUI prompt per shot
    # Models stay loaded between the two shots: one unload/free around the render stage.
    assert hooks == ["unload_ollama", "free_comfyui"]
    wf = next(iter(fake.prompts.values()))
    # Each shot renders with its own compiled prompt: framing first, then subject and look.
    text = wf["5"]["inputs"]["text"]
    assert text.startswith("Cinematic film still, (medium shot:1.3), (eye-level shot:1.25)")
    assert "retro clay sci-fi" in text and "--neg" not in text
    assert "distorted hands" in wf["6"]["inputs"]["text"]  # global negative prompt
    with ctx.db.session() as s:
        renders = s.scalars(select(Render).where(Render.project_id == pid)).all()
        assert all(r.remote_id and r.workflow == "v2v_3070_quality" for r in renders)
        assert "WORKFLOW_COMPILED" in events(ctx, pid)


def test_comfyui_oom_triggers_free_and_degraded_resubmit(ctx, sample_video):
    fake = FakeComfyUI(behaviours=["oom"])
    ctx.settings.render.renderer = "comfyui"
    ctx.settings.comfyui.poll_interval_s = 0
    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=fake.transport())
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    assert fake.freed >= 1 and len(fake.prompts) == 3


def test_unconfigured_remote_profile_is_refused_before_processing(ctx, sample_video):
    ctx.settings.render.renderer = "comfyui"
    with ctx.db.transaction() as s, pytest.raises(ValueError, match="Remote rendering"):
        commands.create_project(s, ProjectCreate(
            name="q", render_profile="HYBRID_MAX", source=SourceIn(local_path=str(sample_video)),
            rights=RightsIn(category=RightsCategory.USER_OWNED), creative=CreativeIn(theme="x"),
            autostart=True), ctx.settings)
    with ctx.db.session() as s:
        assert s.scalar(select(func.count(Job.id))) == 0


@pytest.mark.parametrize("bad_path", ["/nonexistent/video.mp4"])
def test_missing_source_file_fails_without_retry_storm(ctx, bad_path):
    pid = create(ctx, Path(bad_path))
    run(ctx)
    assert status(ctx, pid) == "FAILED"


def test_repairs_stop_when_rerenders_do_not_improve_qc(ctx, sample_video):
    from rokkur_studio.pipeline.stages import _repairs_stalled
    from rokkur_studio.services.events import EventType, record_event
    from rokkur_studio.services.projects import save_document

    pid = create(ctx, sample_video)

    def report(score: float) -> dict[str, Any]:
        return {"shots": [{"shot_id": "s1", "decision": "FAIL", "overall": score},
                          {"shot_id": "s2", "decision": "PASS", "overall": 9.0}]}

    with ctx.db.session() as s:
        for score in (4.0, 4.1):
            save_document(s, pid, "qc_report", report(score), created_by="test")
        assert not _repairs_stalled(s, pid, 3)  # too few rounds to judge
        save_document(s, pid, "qc_report", report(4.15), created_by="test")
        assert _repairs_stalled(s, pid, 3)
        record_event(s, EventType.REPAIR_BUDGET_EXTENDED, project_id=pid, actor="elis",
                     data={"rounds": 5})
        s.flush()
        assert not _repairs_stalled(s, pid, 3)  # granted rounds start a fresh window
        for score in (4.2, 5.0, 6.0):
            save_document(s, pid, "qc_report", report(score), created_by="test")
        assert not _repairs_stalled(s, pid, 3)  # still improving
        save_document(s, pid, "qc_report", report(6.1), created_by="test")
        assert _repairs_stalled(s, pid, 2)  # the default: one round without a gain stops
        assert not _repairs_stalled(s, pid, 3)


def test_timings_report_shows_where_a_render_spent_its_time(ctx, settings, sample_video,
                                                            monkeypatch, capsys):
    from rokkur_studio import cli
    from rokkur_studio.services import timings

    fake = FakeComfyUI()
    ctx.settings.render.renderer = "comfyui"
    ctx.settings.comfyui.poll_interval_s = 0
    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=fake.transport())
    ctx.gpu.after_heavy = []
    pid = create(ctx, sample_video)
    run(ctx)
    with ctx.db.session() as s:
        data = timings.collect(s, pid)
    assert data["stages"]["render"]["runs"] == 1 and data["total_seconds"] > 0
    assert [r["first_in_job"] for r in data["renders"]] == [True, False]
    assert all(r["qc"] is not None for r in data["renders"])  # QC score per attempt
    text = timings.report(data)
    assert "(loads models)" in text and "render" in text and "Shot renders:" in text

    monkeypatch.setattr(cli, "_settings", lambda args: settings)
    assert cli.main(["timings"]) == 0  # the latest project by default
    assert pid in capsys.readouterr().out
    assert cli.main(["timings", "proj_missing"]) == 2
