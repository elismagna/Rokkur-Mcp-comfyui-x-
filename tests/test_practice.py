"""Night School: practice batches on finished videos, general quality scores and lessons."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from rokkur_studio.db.models import Job, Render
from rokkur_studio.services import practice
from rokkur_studio.services import timings as timings_svc
from rokkur_studio.services.projects import budget_usage, get_project, latest_document, save_document
from tests.test_dashboard import client_for
from tests.test_pipeline import create, run, status
from tests.test_quality_stage import SeeingProvider


# -- scoring and picking -------------------------------------------------------------------
def test_quality_score_needs_the_review_and_caps_broken_anatomy():
    plain = practice.quality_score({"overall": 7.2})
    assert plain["score"] == 7.2 and plain["ai"] is None and not plain["reviewed"]
    shot = {"overall": 9.0, "prompt_adherence": 8, "style_consistency": 8, "identity": 8,
            "hand_body_deformation": 8, "picture_review": {"steadiness": 8}}
    assert practice.quality_score(shot)["score"] == 8.5
    melted = {**shot, "hand_body_deformation": 3}
    scored = practice.quality_score(melted)
    assert scored["score"] == 5.0 and scored["anatomy_capped"]


def test_variants_rotate_and_skip_what_the_workflow_or_shot_cannot_change():
    everything = {v.needs: True for v in practice.CATALOG if v.needs}
    first = practice.pick_variants(0, 3, everything, {})
    second = practice.pick_variants(1, 3, everything, {})
    assert first[:2] == [practice.BASELINE, practice.SEED] and len(first) == 5
    assert not set(first[2:]) & set(second[2:])  # the next batch tries other changes
    nothing = practice.pick_variants(0, 20, {}, {"stabilize": "off"})
    assert set(nothing[2:]) == {"stabilizer strong", "smoothed guide"}
    refs = practice.pick_variants(0, 20, {"REFERENCE_IMAGE": True}, {"reference_mode": "none"})
    assert {"reference cutout", "reference first frame"} <= set(refs)
    assert "reference none" not in refs  # what the video already gets
    assert practice.effective_reference("auto", keep=True) == "cutout"
    ov = practice.overrides_for("70% steps", {"cfg": 5.0}, seed=1, steps=20)
    assert ov == {"cfg": 5.0, "steps": 14}
    assert practice.overrides_for(practice.SEED, {}, seed=10, steps=20)["seed"] == 10 + 7919


# -- a real batch ----------------------------------------------------------------------------
def _finished(ctx, sample_video) -> str:
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    return pid


def _queue(ctx, pid: str, batches: int = 1) -> list[str]:
    with ctx.db.transaction() as s:
        jobs = practice.queue_batches(s, ctx.settings, [get_project(s, pid)], batches=batches)
        return [j.id for j in jobs]


def test_a_practice_batch_scores_every_version_and_leaves_the_video_alone(ctx, sample_video):
    ctx.settings.practice.variants = 2
    ctx.extras["vision_provider"] = SeeingProvider()
    pid = _finished(ctx, sample_video)
    with ctx.db.session() as s:
        before = {r.id for r in s.scalars(select(Render).where(Render.project_id == pid))}
        renders_before = budget_usage(s, pid, ctx.settings)["renders"]
    _queue(ctx, pid)
    run(ctx)

    assert status(ctx, pid) == "READY_TO_PUBLISH"  # practice never moves the video
    with ctx.db.session() as s:
        doc = latest_document(s, pid, practice.REPORT)
        assert doc is not None
        report = doc.data
        rows = s.scalars(select(Render).where(Render.project_id == pid,
                                              Render.id.not_in(before))).all()
        assert {r.status for r in rows} == {"practice"}
        assert budget_usage(s, pid, ctx.settings)["renders"] == renders_before
        timed = timings_svc.collect(s, pid)
    labels = [v["label"] for v in report["variants"]]
    assert labels[:2] == [practice.BASELINE, practice.SEED] and len(labels) == 4
    assert report["reviewed"] and report["best"] in labels
    base = report["variants"][0]
    assert base["delta"] == 0 and base["seconds"] is not None  # the seed render took the cold start
    assert report["variants"][1]["seconds"] is None
    assert all(v["sheet_asset_id"] and v["asset_id"] for v in report["variants"])
    assert all(r["status"] != "practice" for r in timed["renders"])
    assert len(timed["renders"]) == len(before)


def test_practice_hands_the_rest_of_a_batch_on_when_real_work_arrives(ctx, sample_video,
                                                                      monkeypatch):
    ctx.settings.practice.variants = 2
    pid = _finished(ctx, sample_video)
    calls = {"n": 0}

    def waiting(session: Any) -> bool:
        calls["n"] += 1
        return calls["n"] == 1  # a real job shows up once, after the first render

    monkeypatch.setattr(practice, "real_work_waiting", waiting)
    first = _queue(ctx, pid)[0]
    run(ctx)
    with ctx.db.session() as s:
        jobs = s.scalars(select(Job).where(Job.kind == practice.KIND)
                         .order_by(Job.created_at)).all()
        doc = latest_document(s, pid, practice.REPORT)
    assert len(jobs) == 2 and jobs[1].payload["continues"] == first
    assert doc is not None and doc.data["batch"] == first
    assert len(doc.data["variants"]) == 4
    assert not doc.data["reviewed"]  # rule-based agents cannot see: not counted as a lesson


def test_reference_variants_render_with_their_own_reference_type(ctx, sample_video):
    pid = _finished(ctx, sample_video)
    labels = [practice.BASELINE, practice.SEED, "reference cutout", "reference first frame"]
    with ctx.db.transaction() as s:
        practice.queue_batches(s, ctx.settings, [get_project(s, pid)], batches=1)
        job = s.scalars(select(Job).where(Job.kind == practice.KIND)).one()
        job.payload = {**job.payload, "variants": labels}
    run(ctx)
    with ctx.db.session() as s:
        report = latest_document(s, pid, practice.REPORT).data  # type: ignore[union-attr]
        manifest = latest_document(s, pid, "manifest").data  # type: ignore[union-attr]
    rows = {v["label"]: v for v in report["variants"]}
    assert rows["reference cutout"]["overrides"] == {"reference_mode": "cutout"}
    assert rows["reference first frame"]["overrides"] == {"reference_mode": "source"}
    assert manifest["identity"]["reference_mode"] != "cutout"  # the video's plan is untouched


def test_practice_needs_a_rendered_video(ctx, sample_video):
    pid = create(ctx, sample_video, autostart=False)
    with ctx.db.transaction() as s, pytest.raises(practice.PracticeError):
        practice.queue_batches(s, ctx.settings, [get_project(s, pid)], batches=1)


# -- lessons -------------------------------------------------------------------------------
def _fake_report(ctx, pid: str, base: float, changes: dict[str, float],
                 seed: float = 0.1) -> None:
    def row(label: str, score: float, seconds: float = 100.0) -> dict[str, Any]:
        return {"label": label, "score": score, "reviewed": True, "seconds": seconds,
                "qc": score, "ai": score, "anatomy": 8, "delta": score - base,
                "asset_id": "a", "sheet_asset_id": "b"}
    variants = [row(practice.BASELINE, base), row(practice.SEED, base + seed)]
    variants += [row(lb, base + d, 70.0 if lb == "70% steps" else 100.0)
                 for lb, d in changes.items()]
    with ctx.db.transaction() as s:
        save_document(s, pid, practice.REPORT, {"profile": "PREVIEW", "shot_id": "shot_001",
                                                "variants": variants, "reviewed": True},
                      created_by="test")


def test_lessons_need_enough_batches_and_a_gain_beyond_the_seed(ctx, sample_video):
    pid = create(ctx, sample_video, autostart=False)
    for base in (6.0, 7.0, 5.5):
        _fake_report(ctx, pid, base, {"cfg 4.5": 1.0, "cfg 7.5": -1.2, "70% steps": 0.1,
                                      "shift 5": 0.3})
    _fake_report(ctx, pid, 6.0, {"control 0.85": 2.0})
    with ctx.db.session() as s:
        learned = practice.lessons(s, ctx.settings)["PREVIEW"]
        proven = practice.proven_overrides(s, ctx.settings, "PREVIEW")
        text = practice.report_text(practice.lessons(s, ctx.settings), practice.reports(s))
    verdicts = {c["label"]: c["verdict"] for c in learned["changes"]}
    assert verdicts == {"cfg 4.5": "better", "cfg 7.5": "worse", "70% steps": "as good, faster",
                        "shift 5": "no clear difference", "control 0.85": "needs 2 more"}
    assert learned["batches"] == 4 and learned["seed_swing"] == 0.1
    assert proven == {"cfg": 4.5, "steps_scale": 0.7}
    assert "cfg 4.5" in text and "better" in text


def test_proven_lessons_start_new_videos_only_when_switched_on(ctx, sample_video):
    other = create(ctx, sample_video, autostart=False)
    for base in (6.0, 7.0, 5.5):
        _fake_report(ctx, other, base, {"cfg 4.5": 1.0, "reference first frame": 1.0})
    ctx.settings.practice.apply_lessons = True
    pid = create(ctx, sample_video, cfg=7.0, reference_mode="none")
    plain = create(ctx, sample_video)
    run(ctx)
    with ctx.db.session() as s:
        own = latest_document(s, pid, "manifest").data  # type: ignore[union-attr]
        learned = latest_document(s, plain, "manifest").data  # type: ignore[union-attr]
    # the video's own choices win; lessons fill in what was left at the form's default
    assert all(sh["overrides"]["cfg"] == 7.0 for sh in own["shots"])
    assert own["identity"]["reference_mode"] == "none"
    assert all(sh["overrides"]["cfg"] == 4.5 for sh in learned["shots"])
    assert all("reference_mode" not in sh["overrides"] for sh in learned["shots"])
    assert learned["identity"]["reference_mode"] == "source"


def test_night_school_page_lists_lessons_and_queues_batches(ctx, sample_video):
    pid = _finished(ctx, sample_video)
    for base in (6.0, 7.0, 5.5):
        _fake_report(ctx, pid, base, {"cfg 4.5": 1.0})
    client = client_for(ctx)
    page = client.get("/ui/practice").text
    assert "Night School" in page and "cfg 4.5" in page and "better" in page
    r = client.post("/ui/practice", data={"project_id": pid, "batches": "2"},
                    follow_redirects=False)
    assert r.status_code == 303 and "Queued%202" in r.headers["location"]
    with ctx.db.session() as s:
        queued = s.scalars(select(Job).where(Job.kind == practice.KIND)).all()
    assert len(queued) == 2 and all(j.project_id is None for j in queued)


def test_practice_and_lessons_commands(ctx, sample_video, monkeypatch, capsys):
    from rokkur_studio import cli

    monkeypatch.setattr(cli, "_settings", lambda args: ctx.settings)
    assert cli.main(["practice"]) == 1  # nothing rendered yet
    pid = _finished(ctx, sample_video)
    assert cli.main(["practice", "--all", "--batches", "3"]) == 0
    assert "Queued 3 practice batches" in capsys.readouterr().out
    with ctx.db.session() as s:
        rounds = [j.payload["round"] for j in s.scalars(select(Job).where(
            Job.kind == practice.KIND).order_by(Job.created_at))]
    assert rounds == [0, 1, 2]
    for base in (6.0, 7.0, 5.5):
        _fake_report(ctx, pid, base, {"cfg 4.5": 1.0})
    assert cli.main(["lessons"]) == 0
    out = capsys.readouterr().out
    assert "PREVIEW: 3 batches" in out and "better" in out
