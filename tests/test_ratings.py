"""Your ratings: storing them, what they change in the pipeline, and what the studio learns."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url

from rokkur_studio.db.models import Base, Rating, Render
from rokkur_studio.manifest.schema import ReconstructionManifest
from rokkur_studio.services import commands, ratings, taste
from rokkur_studio.services.projects import (
    at_repair_limit,
    get_project,
    latest_document,
    save_document,
)
from tests.conftest import TEST_DB
from tests.test_pipeline import create, events, run, status


def rate(ctx, pid: str, target: str, value: int, **kw) -> Rating | None:
    with ctx.db.transaction() as s:
        return ratings.rate(s, get_project(s, pid),
                            ratings.RatingIn(target=target, value=value, **kw), actor="test")


def renders(ctx, pid: str) -> list[tuple[str, int, str]]:
    with ctx.db.session() as s:
        rows = s.scalars(select(Render).where(Render.project_id == pid)
                         .order_by(Render.shot_id, Render.attempt)).all()
        return [(r.shot_id, r.attempt, r.status) for r in rows]


def test_rating_a_shot_freezes_what_made_it_and_can_change_or_be_removed(ctx, sample_video):
    pid = create(ctx, sample_video)
    run(ctx)
    rate(ctx, pid, "shot_001", 2, tags=["style", "motion"], note="  love the light  ")
    rate(ctx, pid, "shot_001", -1, tags=["flicker"])  # changed my mind: same row, new verdict
    with ctx.db.session() as s:
        rows = s.scalars(select(Rating).where(Rating.project_id == pid)).all()
        assert len(rows) == 1
        r = rows[0]
        assert (r.value, r.tags, r.note) == (-1, ["flicker"], None)
        snap = r.snapshot
        assert snap["kind"] == "shot" and snap["shot_id"] == "shot_001" and snap["attempt"] == 1
        assert snap["theme"] == "retro clay sci-fi" and snap["prompt"]
        assert snap["seed"] is not None and snap["profile"] == "PREVIEW"
        assert snap["qc"]["decision"] == "PASS" and snap["qc"]["overall"] is not None
    rate(ctx, pid, "shot_001", 0)
    with ctx.db.session() as s:
        assert s.scalar(select(Rating.id).where(Rating.project_id == pid)) is None
    assert "RATING_SET" in events(ctx, pid) and "RATING_CLEARED" in events(ctx, pid)


def test_ratings_are_validated(ctx, sample_video):
    with pytest.raises(ValueError, match="unknown tags"):
        ratings.RatingIn(target="shot_001", value=1, tags=["vibes"])
    with pytest.raises(ValueError):
        ratings.RatingIn(target="shot_001", value=3)
    with pytest.raises(ValueError):
        ratings.RatingIn(target="anything", value=1)
    pid = create(ctx, sample_video, autostart=False)
    with pytest.raises(ValueError, match="no video to rate"):
        rate(ctx, pid, "video", 1)
    with pytest.raises(ValueError, match="no finished render"):
        rate(ctx, pid, "shot_001", 1)


def test_rating_the_whole_video(ctx, sample_video):
    pid = create(ctx, sample_video)
    run(ctx)
    r = rate(ctx, pid, "video", 2, tags=["style"])
    assert r is not None and r.asset_id and r.render_id is None
    assert r.snapshot["kind"] == "video" and r.snapshot["asset_kind"] == "final"
    assert r.snapshot["profile"] == "PREVIEW" and r.snapshot["qc"]["decision"] == "PASS"


def test_a_render_you_like_passes_quality_check(ctx, sample_video):
    ctx.settings.render.max_retries = 1
    pid = create(ctx, sample_video,
                 test_faults={"shot_002": {"kind": "black", "attempts": [1, 2, 3]}})
    run(ctx)
    assert status(ctx, pid) == "FAILED"
    rate(ctx, pid, "shot_002", 1, tags=["style"])  # the black frames are what I wanted
    with ctx.db.transaction() as s:
        commands.recheck_quality(s, get_project(s, pid, for_update=True), ctx.settings,
                                 actor="test")
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.session() as s:
        qc = latest_document(s, pid, "qc_report").data
        shot = next(x for x in qc["shots"] if x["shot_id"] == "shot_002")
        assert shot["decision"] == "PASS" and shot["accepted_by"] == "you"
        assert shot["issues"]  # what QC measured is still on record


def test_redo_disliked_shots_on_a_finished_video(ctx, sample_video):
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.transaction() as s:
        meta = latest_document(s, pid, "metadata").data
        save_document(s, pid, "metadata", {**meta, "title": "My own title #shorts"},
                      created_by="dashboard")
        before = ReconstructionManifest.model_validate(latest_document(s, pid, "manifest").data)
        first_seed = before.shot("shot_001").seed
    rate(ctx, pid, "shot_001", -2, tags=["style"])
    with ctx.db.transaction() as s:
        commands.redo_shots(s, get_project(s, pid, for_update=True), ctx.settings,
                            shot_ids=["shot_001"], actor="test")
        assert get_project(s, pid).status == "RENDER_QUEUED"
        manifest = ReconstructionManifest.model_validate(
            latest_document(s, pid, "manifest").data)
        changes = manifest.shot("shot_001").overrides
        assert changes["seed"] != first_seed and changes["control_strength"] == 0.9
        assert manifest.shot("shot_002").overrides == before.shot("shot_002").overrides
        plan = latest_document(s, pid, "repair_plan").data
        assert plan["requested_by"] == "test" and plan["actions"][0]["tags"] == ["style"]
        assert "more freedom for the new look" in plan["actions"][0]["reason"]
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    assert renders(ctx, pid) == [("shot_001", 1, "superseded"), ("shot_001", 2, "succeeded"),
                                 ("shot_002", 1, "succeeded")]
    with ctx.db.session() as s:
        assert latest_document(s, pid, "metadata").data["title"] == "My own title #shorts"
    assert "SHOTS_REDO_REQUESTED" in events(ctx, pid)
    with ctx.db.transaction() as s:  # a second redo keeps your text too
        commands.redo_shots(s, get_project(s, pid, for_update=True), ctx.settings,
                            shot_ids=["shot_002"], actor="test")
    run(ctx)
    with ctx.db.session() as s:
        assert latest_document(s, pid, "metadata").data["title"] == "My own title #shorts"


def test_redo_at_the_repair_limit_keeps_the_shots_you_did_not_pick(ctx, sample_video):
    ctx.settings.render.max_retries = 1
    pid = create(ctx, sample_video,
                 test_faults={"shot_002": {"kind": "black", "attempts": [1, 2]},
                              "shot_001": {"kind": "flicker", "attempts": [1, 2]}})
    run(ctx)
    with ctx.db.session() as s:
        assert at_repair_limit(get_project(s, pid))
    with ctx.db.transaction() as s:  # redo only shot 2; shot 1's flicker stays as it is
        commands.redo_shots(s, get_project(s, pid, for_update=True), ctx.settings,
                            shot_ids=["shot_002"], actor="test")
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    attempts = renders(ctx, pid)
    assert ("shot_001", 3, "succeeded") not in attempts  # never sent back for repair
    assert ("shot_002", 3, "succeeded") in attempts
    with ctx.db.session() as s:
        qc = latest_document(s, pid, "qc_report").data
        assert {x["shot_id"]: x.get("accepted_by") for x in qc["shots"]}["shot_001"] == "you"


def test_redo_is_refused_while_rendering_and_for_unknown_shots(ctx, sample_video):
    pid = create(ctx, sample_video, autostart=False)
    with ctx.db.transaction() as s, pytest.raises(ValueError, match="finished video"):
        commands.redo_shots(s, get_project(s, pid, for_update=True), ctx.settings,
                            shot_ids=["shot_001"], actor="test")
    pid = create(ctx, sample_video)
    run(ctx)
    with ctx.db.transaction() as s, pytest.raises(ValueError, match="Unknown shots"):
        commands.redo_shots(s, get_project(s, pid, for_update=True), ctx.settings,
                            shot_ids=["shot_009"], actor="test")
    with ctx.db.transaction() as s, pytest.raises(ValueError, match="at least one"):
        commands.redo_shots(s, get_project(s, pid, for_update=True), ctx.settings,
                            shot_ids=[], actor="test")


def test_redo_changes_follow_your_tags_within_bounds():
    manifest = ReconstructionManifest.model_validate({
        "project_id": "p", "source_asset": "p/src.mp4", "render_profile": "PREVIEW",
        "video": {"fps": 24, "width": 640, "height": 360, "duration": 4},
        "style": {"theme": "t", "prompt": "t"},
        "shots": [{"shot_id": "shot_001", "start": 0, "end": 4, "seed": 5,
                   "overrides": {"control_strength": 0.95}}]})
    firmer, why = ratings.redo_changes(manifest, "shot_001", ["motion"], attempt=1)
    assert firmer["control_strength"] == 1.0 and "follows the source more closely" in why[1]
    looser, _ = ratings.redo_changes(manifest, "shot_001", ["prompt"], attempt=1)
    assert looser["control_strength"] == 0.85
    both, _ = ratings.redo_changes(manifest, "shot_001", ["motion", "style"], attempt=1)
    assert "control_strength" not in both
    seed_only, why = ratings.redo_changes(manifest, "shot_001", ["flicker"], attempt=2)
    assert set(seed_only) == {"seed"} and "no automatic fix yet" in why[-1]
    unsupported, _ = ratings.redo_changes(manifest, "shot_001", ["motion"], attempt=1,
                                          supported={"SEED"})
    assert set(unsupported) == {"seed"}
    assert firmer["seed"] != seed_only["seed"]


def test_prompt_terms_drop_weights_vocabulary_and_long_phrases():
    terms = taste.prompt_terms("Stop-motion film still, (close-up shot:1.3), (low-angle shot:1.2), "
                               "golden hour diffusion, (Warm Clay:1.1), a very long description "
                               "of a scene that goes on, cel shading")
    assert terms == ["stop-motion film still", "warm clay", "cel shading"]


def _seed_ratings(ctx, sample_video, n: int, theme: str, value: int, *, cs: float = 1.0) -> None:
    pid = create(ctx, sample_video, autostart=False)
    with ctx.db.transaction() as s:
        for i in range(n):
            s.add(Rating(project_id=pid, target=f"shot_{i:03d}", value=value, tags=["style"],
                         created_by="test", snapshot={
                             "kind": "shot", "prompt": f"medium shot, {theme}, film grain",
                             "profile": "RTX3070_QUALITY", "workflow": "v2v_3070_quality",
                             "control_strength": cs, "subject_mode": "keep",
                             "qc": {"decision": "PASS" if value > 0 else "FAIL",
                                    "overall": 7.0 if value > 0 else 4.0}}))


def test_taste_learns_across_projects_and_only_suggests_with_evidence(ctx, sample_video):
    with ctx.db.session() as s:
        assert taste.build_profile(s)["count"] == 0
    _seed_ratings(ctx, sample_video, 3, "claymation", 2, cs=0.85)
    with ctx.db.session() as s:
        one = taste.build_profile(s)
    assert one["projects"] == 1 and all(x["confidence"] == "early" for x in one["liked"])
    assert taste.suggestions(one) == []  # one project is one example, not a rule
    _seed_ratings(ctx, sample_video, 1, "claymation", 2, cs=0.85)
    _seed_ratings(ctx, sample_video, 2, "neon noir", -2)
    _seed_ratings(ctx, sample_video, 2, "neon noir", -1)
    with ctx.db.session() as s:
        profile = taste.build_profile(s)
    assert profile["count"] == 8 and profile["projects"] == 4
    liked = {x["value"]: x for x in profile["liked"]}
    disliked = {x["value"]: x for x in profile["disliked"]}
    assert liked["claymation"]["projects"] == 2 and liked["claymation"]["lift"] > 0.4
    assert disliked["neon noir"]["lift"] < -0.4
    assert "film grain" not in liked and "film grain" not in disliked  # everywhere: no signal
    assert "medium shot" not in liked  # framing vocabulary is not a prompt term
    tips = {(t["field"], t["value"]) for t in taste.suggestions(profile)}
    assert ("theme", "claymation") in tips and ("negative_prompt", "neon noir") in tips
    assert ("control_strength", "0.85") in tips
    assert profile["qc"]["percent"] == 100 and profile["tags"]["liked"]["Style"] == 4
    text_report = taste.report(profile)
    assert "claymation" in text_report and "QC agreed with you on 8 of 8" in text_report


def test_human_ratings_only(ctx, sample_video):
    _seed_ratings(ctx, sample_video, 2, "claymation", 2)
    with ctx.db.transaction() as s:
        for r in s.scalars(select(Rating)):
            r.rater = "ai"
    with ctx.db.session() as s:
        assert taste.build_profile(s)["count"] == 0


def test_migrations_build_the_same_schema_as_the_models(database):
    """alembic upgrade head on an empty database matches the SQLAlchemy models."""
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    from tests.conftest import ROOT

    url = make_url(TEST_DB)
    name = (url.database or "").removesuffix("_test") + "_migrations_test"
    try:
        with database.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # pragma: no cover - a test role without CREATEDB
        pytest.skip(f"cannot create a scratch database: {exc}")
    target = url.set(database=name).render_as_string(hide_password=False)
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["url"] = target
    command.upgrade(cfg, "head")
    engine = create_engine(target)
    try:
        with engine.connect() as c:
            diff = compare_metadata(MigrationContext.configure(
                c, opts={"compare_type": True}), Base.metadata)
        assert diff == []
    finally:
        engine.dispose()
        with database.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
