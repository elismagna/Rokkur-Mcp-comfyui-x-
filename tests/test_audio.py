"""Sound: generation through a fake ComfyUI, FFmpeg edits, soundtracks at any stage."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from rokkur_studio.comfyui.compiler import TemplateRegistry, compile_workflow
from rokkur_studio.db.models import Asset, AudioClip, CostEntry, Event, GpuLease, Job
from rokkur_studio.domain.states import InvalidTransition
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.services import audio as svc
from rokkur_studio.services import commands
from rokkur_studio.services.audio import AudioEdit, AudioRequest, AudioStore
from rokkur_studio.services.projects import get_project
from tests.conftest import ROOT
from tests.test_dashboard import client_for
from tests.test_images import with_comfy
from tests.test_pipeline import create, run, status


def store_for(ctx) -> AudioStore:
    return AudioStore(ctx.settings.studio.data_dir)


def request(ctx, **fields) -> list[AudioClip]:
    with ctx.db.transaction() as s:
        rows = svc.request_audio(s, ctx.settings, store_for(ctx), AudioRequest(**fields))
        return [s.get(AudioClip, r.id) for r in rows]


def edit(ctx, **fields) -> AudioClip:
    with ctx.db.transaction() as s:
        clip = svc.edit_audio(s, store_for(ctx), ctx.ffmpeg, AudioEdit(**fields))
        s.refresh(clip)
        return clip


def clips(ctx, ids: list[str]) -> list[AudioClip]:
    with ctx.db.session() as s:
        return [s.get(AudioClip, i) for i in ids]


def tone(ctx, path: Path, seconds: float = 2.0, frequency: int = 440) -> AudioClip:
    ctx.ffmpeg.make_test_audio(path, seconds=seconds, frequency=frequency)
    with ctx.db.transaction() as s:
        clip = svc.import_audio(s, store_for(ctx), ctx.ffmpeg, path, title=path.stem,
                                request={"rights_evidence": "a test tone"})
        s.refresh(clip)
        return clip


def length(ctx, clip: AudioClip) -> float:
    return ctx.ffmpeg.audio_info(store_for(ctx).path_for(clip.rel_path)).duration


def test_the_audio_templates_compile_and_the_profiles_load(ctx):
    registry = TemplateRegistry(ROOT / "workflows")
    music = compile_workflow(registry.get("audio_ace_step"), {
        "TAGS": "ambient, piano", "LYRICS": "[inst]", "SECONDS": 20, "SEED": 1, "STEPS": 10,
        "CFG": 5, "BATCH": 2, "OUTPUT_PREFIX": "rokkur/audio/t"})
    latent = next(n for n in music.workflow.values() if n["class_type"] == "EmptyAceStepLatentAudio")
    assert latent["inputs"] == {"seconds": 20, "batch_size": 2}
    assert registry.get("audio_ace_step").spec.output_kinds == ["audio"]
    sound = compile_workflow(registry.get("audio_stable_open"), {
        "PROMPT": "rain", "NEGATIVE_PROMPT": "music", "SECONDS": 12, "SEED": 1, "STEPS": 10,
        "CFG": 4, "BATCH": 1, "OUTPUT_PREFIX": "rokkur/audio/t"})
    cond = next(n for n in sound.workflow.values() if n["class_type"] == "ConditioningStableAudio")
    assert cond["inputs"]["seconds_total"] == 12 and sound.workflow["6"]["inputs"]["seconds"] == 12
    assert set(ctx.settings.audio_profiles) == {"ACE_STEP", "STABLE_AUDIO"}
    assert ctx.settings.audio_profile("STABLE_AUDIO").max_seconds == 47
    assert "ComfyUI renderer" in ctx.settings.audio_profile_problem("ACE_STEP")
    with_comfy(ctx)
    assert ctx.settings.audio_profile_problem("ACE_STEP") is None
    assert "unknown audio profile" in ctx.settings.audio_profile_problem("NOPE")


def test_generation_needs_the_renderer_a_prompt_and_a_length_the_model_can_make(ctx):
    with pytest.raises(ValueError, match="ComfyUI renderer"):
        request(ctx, prompt="piano")
    with_comfy(ctx)
    with pytest.raises(ValueError, match="Describe the sound"):
        request(ctx, prompt="  ")
    with pytest.raises(ValueError, match="at most 47"):
        request(ctx, operation="sound", prompt="rain", seconds=90)
    with pytest.raises(ValueError, match="makes sound, not music"):
        request(ctx, operation="music", prompt="piano", profile="STABLE_AUDIO")


def test_music_and_effects_are_made_through_comfyui_with_a_gpu_lease(ctx):
    fake = with_comfy(ctx)
    made = request(ctx, prompt="lo-fi, warm, 70 bpm", count=2, seconds=4, seed=11,
                   title="Bed", lyrics="[verse]\nhello")
    assert [m.status for m in made] == ["queued"] * 2 and {m.seed for m in made} == {11}
    assert made[0].params["LYRICS"] == "[verse]\nhello" and made[0].workflow == "audio_ace_step"
    assert Worker(ctx, worker_id="t").drain() == 1
    done = clips(ctx, [m.id for m in made])
    assert [d.status for d in done] == ["done", "done"]
    assert all(d.rel_path and d.rel_path.endswith("clip.flac") for d in done)
    assert all(abs((d.duration_s or 0) - 4.0) < 0.1 and d.sample_rate == 8000 for d in done)
    assert ctx.ffmpeg.audio_info(store_for(ctx).path_for(done[0].rel_path)).codec == "flac"
    sent = next(iter(fake.prompts.values()))
    assert sent["14"]["inputs"]["tags"] == "lo-fi, warm, 70 bpm"
    assert sent["17"]["inputs"]["batch_size"] == 2
    assert sent["60"]["inputs"]["filename_prefix"].startswith("rokkur/audio/")
    with ctx.db.session() as s:
        costs = s.scalars(select(CostEntry).where(CostEntry.job_id == done[0].job_id)).all()
        assert len(costs) == 2 and {c.kind for c in costs} == {"gpu_minutes"}
        leases = s.scalars(select(GpuLease)).all()
        assert leases and all(lease.released_at is not None for lease in leases)
    effect = request(ctx, operation="sound", prompt="rain on a tin roof", negative_prompt="music",
                     seconds=6)[0]
    assert effect.profile == "STABLE_AUDIO" and effect.params["NEGATIVE_PROMPT"] == "music"
    assert effect.prompt == "rain on a tin roof" and effect.lyrics == ""
    Worker(ctx, worker_id="t").drain()
    effect = clips(ctx, [effect.id])[0]
    assert effect.status == "done" and abs(effect.duration_s - 6.0) < 0.1
    with ctx.db.session() as s:
        assert {lease.resource_class for lease in s.scalars(select(GpuLease))} == {"GPU_HEAVY", "GPU_MEDIUM"}


def test_oom_walks_the_audio_ladder_down_to_one_shorter_clip(ctx):
    fake = with_comfy(ctx, ["oom", "oom", "oom", "ok", "ok"])
    made = request(ctx, prompt="drums", count=2, seconds=40, seed=3)
    Worker(ctx, worker_id="t").drain()
    done = clips(ctx, [m.id for m in made])
    assert [d.status for d in done] == ["done", "done"]
    assert done[0].params["_oom_steps"] == ["clear_cache", "single_clip", "shorter"]
    assert done[0].params["SECONDS"] == 20 and abs(done[1].duration_s - 20) < 0.1
    assert fake.freed >= 1 and len(fake.prompts) == 5
    seeds = sorted(n["inputs"]["seed"] for p in list(fake.prompts.values())[3:]
                   for n in p.values() if n["class_type"] == "KSampler")
    assert seeds == [3, 4]
    fake.behaviours = ["node_error"]
    failed = request(ctx, prompt="x")[0]
    Worker(ctx, worker_id="t").drain()
    assert clips(ctx, [failed.id])[0].error["code"] == "render_rejected"


def test_every_edit_makes_a_new_clip_from_the_old_one(ctx, tmp_path, sample_video):
    a = tone(ctx, tmp_path / "a.wav", 2.0)
    b = tone(ctx, tmp_path / "b.flac", 1.0, 880)
    assert a.kind == "upload" and a.status == "done" and a.channels == 2 and a.duration_s == 2.0
    trimmed = edit(ctx, operation="trim", source_id=a.id, start=0.5, end=1.5)
    assert trimmed.parent_id == a.id and abs(length(ctx, trimmed) - 1.0) < 0.05
    faded = edit(ctx, operation="fade", source_id=a.id, fade_in=0.2, fade_out=0.4)
    assert faded.kind == "fade" and abs(length(ctx, faded) - 2.0) < 0.05
    quiet = edit(ctx, operation="gain", source_id=a.id, db=-12)
    assert quiet.request["db"] == -12
    level = edit(ctx, operation="normalize", source_id=a.id, lufs=-16)
    assert level.kind == "normalize"
    looped = edit(ctx, operation="loop", source_id=b.id, seconds=4.5, crossfade=0.3)
    assert abs(length(ctx, looped) - 4.5) < 0.05
    fast = edit(ctx, operation="speed", source_id=a.id, factor=2.0)
    assert abs(length(ctx, fast) - 1.0) < 0.05
    slow = edit(ctx, operation="speed", source_id=a.id, factor=0.5, keep_pitch=False)
    assert abs(length(ctx, slow) - 4.0) < 0.1
    mixed = edit(ctx, operation="mix", source_ids=[a.id, b.id], volumes=[1, 0.5], offsets=[0, 1.5])
    assert mixed.parent_id == a.id and abs(length(ctx, mixed) - 2.5) < 0.1
    joined = edit(ctx, operation="concat", source_ids=[a.id, b.id])
    assert abs(length(ctx, joined) - 3.0) < 0.05
    with pytest.raises(ValueError, match="Confirm"):
        edit(ctx, operation="extract", source_path=str(sample_video))
    taken = edit(ctx, operation="extract", source_path=str(sample_video), rights_confirmed=True,
                 rights_evidence="my clip")
    assert taken.kind == "extract" and abs(length(ctx, taken) - 4.0) < 0.15
    with pytest.raises(ValueError, match="after the start"):
        edit(ctx, operation="trim", source_id=a.id, start=1.5, end=1.0)
    with pytest.raises(ValueError, match="two or more"):
        edit(ctx, operation="mix", source_ids=[a.id])
    with ctx.db.session() as s:
        assert s.scalars(select(AudioClip).where(AudioClip.status != "done")).all() == []
        kids = s.scalars(select(AudioClip).where(AudioClip.parent_id == a.id)).all()
        assert {k.kind for k in kids} >= {"trim", "fade", "gain", "normalize", "speed", "mix", "concat"}
    with ctx.db.transaction() as s:
        svc.set_verdict(s, svc.get_clip(s, a.id), 1)
        svc.delete_clip(s, store_for(ctx), svc.get_clip(s, a.id))
    with ctx.db.session() as s:
        assert s.get(AudioClip, a.id) is None and s.get(AudioClip, trimmed.id).parent_id is None
    assert not (store_for(ctx).root / a.id).exists()


def test_a_soundtrack_can_be_set_before_the_edit_stage_and_changed_on_a_finished_video(
        ctx, tmp_path, sample_video):
    bed = tone(ctx, tmp_path / "bed.wav", 1.0, 660)
    bed_path = str(store_for(ctx).path_for(bed.rel_path))
    pid = create(ctx, sample_video)
    Worker(ctx, worker_id="t").drain(max_jobs=3)  # rights, ingest, analyze
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        result = commands.set_soundtrack(s, project, ctx.settings, actor="t", bed_path=bed_path,
                                         rights=svc.rights_line(bed), clip_id=bed.id, gain=0.4,
                                         keep_source_audio=False)
        assert result["re_edit"] is False and result["job_id"] is None
        creative = project.creative_input
        assert creative["audio_bed_path"] == bed_path and creative["audio_bed_gain"] == 0.4
        assert creative["keep_source_audio"] is False and creative["audio_bed_rights_confirmed"]
        assert "clip " + bed.id in creative["audio_bed_rights_evidence"]
        with pytest.raises(ValueError, match="where the soundtrack comes from"):
            commands.set_soundtrack(s, project, ctx.settings, actor="t", bed_path=bed_path, rights="")
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.session() as s:
        finals = s.scalars(select(Asset).where(Asset.project_id == pid, Asset.kind == "final")
                           .order_by(Asset.created_at)).all()
        assert len(finals) == 1
        first = ctx.ffmpeg.probe(ctx.store.path_for(finals[0].rel_path))
        assert first.has_audio and abs(first.duration - 4.0) < 0.2  # the 1 s bed looped under it
    # change it on the finished video: the edit runs again and a new final appears
    other = tone(ctx, tmp_path / "other.wav", 3.0, 220)
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        result = commands.set_soundtrack(
            s, project, ctx.settings, actor="t", bed_path=str(store_for(ctx).path_for(other.rel_path)),
            rights=svc.rights_line(other), clip_id=other.id, gain=0.5, keep_source_audio=True)
        assert result["re_edit"] is True and project.status == "EDITING"
        job = s.get(Job, result["job_id"])
        assert job.kind == "edit" and job.status == "QUEUED"
    Worker(ctx, worker_id="t").drain()
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.session() as s:
        finals = s.scalars(select(Asset).where(Asset.project_id == pid, Asset.kind == "final")
                           .order_by(Asset.created_at)).all()
        assert len(finals) == 2
        kinds = [e.type for e in s.scalars(select(Event).where(Event.project_id == pid))]
        assert kinds.count("SOUNDTRACK_SET") == 2 and kinds.count("FINAL_ENCODED") == 2
        project = get_project(s, pid)
        assert project.creative_input["audio_clip_id"] == other.id
    # removing the added track also re-edits; a published video keeps its sound
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        result = commands.set_soundtrack(s, project, ctx.settings, actor="t", bed_path=None, rights=None)
        assert result["re_edit"] and "audio_bed_path" not in project.creative_input
    Worker(ctx, worker_id="t").drain()
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        project.status = "PUBLISHED"
    with ctx.db.transaction() as s, pytest.raises(InvalidTransition, match="keeps its sound"):
        commands.set_soundtrack(s, get_project(s, pid, for_update=True), ctx.settings, actor="t",
                                bed_path=bed_path, rights="x")


def test_sound_page_api_and_cli_cover_the_whole_flow(ctx, tmp_path, sample_video, monkeypatch,
                                                      capsys):
    from rokkur_studio import cli
    from rokkur_studio.pipeline import context as context_mod

    fake = with_comfy(ctx)
    c = client_for(ctx)
    page = c.get("/ui/audio").text
    assert "No sound yet" in page and "Make a sound" in page and "Shape a clip" in page
    r = c.post("/ui/audio", data={"operation": "music", "prompt": "synthwave, 110 bpm",
                                  "seconds": "8", "count": "1", "seed": "4"}, follow_redirects=False)
    assert r.status_code == 303 and "clip=aud_" in r.headers["location"]
    first = r.headers["location"].split("clip=")[1].split("&")[0].split("#")[0]
    assert "queued" in c.get(f"/ui/audio?clip={first}").text
    Worker(ctx, worker_id="t").drain()
    page = c.get(f"/ui/audio?clip={first}").text
    assert "<code>4</code>" in page and "Use as soundtrack" in page and "8.0 s" in page
    assert c.get(f"/audio/{first}/file").headers["content-type"].startswith("audio/flac")
    wave = c.get(f"/audio/{first}/waveform.png")
    assert wave.status_code == 200 and wave.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert c.post(f"/ui/audio/{first}/verdict", data={"value": "1"},
                  headers={"Accept": "application/json"}).json() == {"verdict": 1}
    # an upload, then edits through the form
    tone_file = ctx.ffmpeg.make_test_audio(tmp_path / "mine.wav", seconds=2)
    with tone_file.open("rb") as fh:
        r = c.post("/ui/audio/upload", files={"source_file": ("mine.wav", fh, "audio/wav")},
                   data={"rights_confirmed": "true", "rights_evidence": "my recording",
                         "title": "Mine"}, follow_redirects=False)
    assert r.status_code == 303 and "clip=aud_" in r.headers["location"]
    mine = r.headers["location"].split("clip=")[1].split("&")[0].split("#")[0]
    r = c.post("/ui/audio/edit", data={"operation": "fade", "source_id": mine, "fade_in": "0.5",
                                       "fade_out": "0.5"}, follow_redirects=False)
    assert r.status_code == 303 and "msg=Fade%20done" in r.headers["location"]
    r = c.post("/ui/audio/edit", data={"operation": "mix", "source_ids": [mine, first],
                                       "volumes": "1, 0.3"}, follow_redirects=False)
    assert "msg=Mix%20done" in r.headers["location"]
    r = c.post("/ui/audio/edit", data={"operation": "trim", "source_id": mine, "start": "3",
                                       "end": "1"}, follow_redirects=False)
    assert "err=The%20edit%20failed" in r.headers["location"]
    # the soundtrack from the page and the API, at the New video stage and on a finished video
    pid = create(ctx, sample_video, autostart=False)
    r = c.post(f"/ui/projects/{pid}/sound", data={"clip_id": mine, "gain": "0.3"},
               follow_redirects=False)
    assert r.status_code == 303 and "Soundtrack%20set" in r.headers["location"]
    page = c.get(f"/ui/projects/{pid}").text
    assert "Mine" in page and "used when the video gets there" in page
    assert "Make music or effects for this video" in page
    ctx.settings.render.renderer = "ffmpeg_preview"
    with ctx.db.transaction() as s:
        commands.start(s, get_project(s, pid, for_update=True), ctx.settings)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    r = c.post(f"/projects/{pid}/soundtrack", json={"clip_id": first, "gain": 0.2})
    assert r.status_code == 200 and r.json()["re_edit"] is True
    assert c.post("/projects/nope/soundtrack", json={"clip_id": first}).status_code == 404
    assert c.post(f"/projects/{pid}/soundtrack", json={"clip_id": first}).status_code == 409  # EDITING
    run(ctx)
    page = c.get(f"/ui/projects/{pid}").text
    assert "edits the finished video again" in page and "Take the final cut" in page
    r = c.post(f"/ui/projects/{pid}/sound/extract", data={"which": "final"}, follow_redirects=False)
    assert r.status_code == 303 and "clip=aud_" in r.headers["location"]
    with ctx.db.session() as s:
        taken = s.scalars(select(AudioClip).where(AudioClip.kind == "extract")).one()
        assert taken.project_id == pid and taken.duration_s > 3.5
    assert c.post(f"/ui/projects/{pid}/sound", data={"remove": "true"},
                  follow_redirects=False).status_code == 303
    # New video takes a library clip as the soundtrack
    page = c.get("/ui/new").text
    assert "Soundtrack from the sound library" in page and "Mine" in page
    # the API
    ctx.settings.render.renderer = "comfyui"
    r = c.post("/audio", json={"operation": "sound", "prompt": "wind", "seconds": 5})
    assert r.status_code == 201 and r.json()[0]["kind"] == "sound"
    assert c.post("/audio", json={"operation": "music", "prompt": ""}).status_code == 422
    r = c.post("/audio/edit", json={"operation": "gain", "source_id": mine, "db": -3})
    assert r.status_code == 201 and r.json()["parent_id"] == mine
    listed = c.get("/audio?limit=3").json()
    assert listed[0]["id"] == r.json()["id"] and listed[0]["file_url"].endswith("/file")
    assert c.get("/audio/profiles").json()["ACE_STEP"]["kind"] == "music"
    assert c.delete(f"/audio/{r.json()['id']}").status_code == 204
    assert c.get(f"/audio/{r.json()['id']}").status_code == 404
    # the CLI
    Worker(ctx, worker_id="t").drain()  # the API's "wind" clip, so --wait makes the CLI's own
    monkeypatch.setattr(cli, "_settings", lambda args: ctx.settings)
    monkeypatch.setattr(context_mod, "build_context", lambda settings, db=None: ctx)
    assert cli.main(["audio", "piano, calm", "--seconds", "3", "--seed", "2", "--wait"]) == 0
    out = capsys.readouterr().out
    assert "queued 1 clip" in out and "done" in out
    assert cli.main(["audio-edit", "loop", mine, "--seconds", "5", "--title", "Looped"]) == 0
    assert "loop" in capsys.readouterr().out
    assert cli.main(["audio-list"]) == 0
    assert "Looped" in capsys.readouterr().out
    assert cli.main(["audio-edit", "trim", "/nowhere.wav", "--rights", "x"]) == 1
    assert fake.prompts
