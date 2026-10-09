"""``rokkur-studio`` command line: migrate, api, worker, comfy-check, agent-check, audit,
smoke-test, render, prompt-schedule."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from rokkur_studio.config import Settings, load_settings
from rokkur_studio.logging_setup import configure_logging


def _settings(args: argparse.Namespace) -> Settings:
    settings = load_settings(Path(args.config) if args.config else None)
    configure_logging(settings.studio.log_level, settings.studio.log_json)
    return settings


def cmd_migrate(args: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    settings = _settings(args)
    # Source checkout: repo root is two levels above the package. Installed (Docker image): the
    # package lives in site-packages, so fall back to the working directory (/app).
    candidates = [Path(__file__).resolve().parents[2], Path.cwd()]
    root = next((c for c in candidates if (c / "migrations" / "env.py").exists()), None)
    if root is None:
        print("migrations/ not found next to the package or in the working directory", file=sys.stderr)
        return 1
    ini = root / "alembic.ini"
    cfg = Config(str(ini)) if ini.exists() else Config()
    cfg.set_main_option("script_location", str(root / "migrations"))
    cfg.attributes["url"] = settings.database.url
    command.upgrade(cfg, args.revision)
    print(f"database migrated to {args.revision}")
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    import uvicorn

    from rokkur_studio.api.app import create_app

    settings = _settings(args)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_config=None)
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    import signal

    from rokkur_studio.jobs.worker import Worker
    from rokkur_studio.pipeline.context import build_context

    settings = _settings(args)
    worker = Worker(build_context(settings), kinds=args.kinds.split(",") if args.kinds else None)
    signal.signal(signal.SIGTERM, lambda *_: worker.stop())
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        worker.stop()
    return 0


def cmd_comfy_check(args: argparse.Namespace) -> int:
    from rokkur_studio.comfyui.client import ComfyClient, ComfyError
    from rokkur_studio.comfyui.compiler import (
        TemplateError,
        TemplateRegistry,
        validate_against_object_info,
    )

    settings = _settings(args)
    url = settings.comfyui.url
    client = ComfyClient(url)
    if getattr(args, "cloud", False):
        if not settings.cloud.ready:
            print("Cloud rendering is not set up: set STUDIO_CLOUD__ENABLED=true and "
                  "STUDIO_CLOUD__URL in .env (docs/cloud.md).")
            return 1
        url = settings.cloud.url
        client = ComfyClient(url, timeout_s=settings.cloud.timeout_s,
                             headers=settings.cloud.headers())
    ok = True
    try:
        stats = client.system_stats()
        for dev in stats.get("devices", []):
            print(f"GPU: {dev.get('name')}  VRAM total {dev.get('vram_total', 0) / 2**30:.1f} GB"
                  f"  free {dev.get('vram_free', 0) / 2**30:.1f} GB")
        info = client.object_info()
        print(f"ComfyUI reachable at {url}: {len(info)} node classes")
    except (ComfyError, httpx.HTTPError) as exc:
        print(f"ComfyUI NOT reachable at {url}: {exc}")
        return 1
    finally:
        client.close()
    registry = TemplateRegistry(settings.workflows_dir)
    needed = ({p.workflow for p in settings.profiles.values()}
              | {p.keep_workflow for p in settings.profiles.values() if p.keep_workflow})
    for name in sorted(set(registry.names()) | needed):
        try:
            template = registry.get(name)
        except TemplateError as exc:
            print(f"  [missing] {name}: {exc}")
            ok = ok and name not in needed
            continue
        problems = validate_against_object_info(template, info)
        unused = "" if name in needed else "  (not used by any render profile)"
        print(f"  [{'ok' if not problems else 'FAIL'}] {name} v{template.spec.version}{unused}")
        for p in problems:
            print(f"      - {p}")
        ok = ok and (not problems or name not in needed)
    return 0 if ok else 1


# Loader node -> the input whose choices are the model files ComfyUI can see.
MODEL_LOADERS = {
    "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
    "diffusion_models": ("UNETLoader", "unet_name"),
    "vae": ("VAELoader", "vae_name"),
    "text_encoders": ("CLIPLoader", "clip_name"),
    "clip_vision": ("CLIPVisionLoader", "clip_name"),
    "loras": ("LoraLoaderModelOnly", "lora_name"),
    "controlnet": ("ControlNetLoader", "control_net_name"),
    "upscale_models": ("UpscaleModelLoader", "model_name"),
}


def comfy_model_files(object_info: dict[str, Any]) -> dict[str, Any]:
    """List the model files each core loader offers, read from ``/object_info`` combo inputs."""
    out: dict[str, Any] = {}
    for kind, (node, field) in MODEL_LOADERS.items():
        spec = object_info.get(node, {}).get("input", {}).get("required", {}).get(field)
        if not spec:
            out[kind] = None  # loader node not installed
            continue
        choices = spec[0]
        if choices == "COMBO" and len(spec) > 1:  # newer schema: ["COMBO", {"options": [...]}]
            choices = spec[1].get("options", [])
        out[kind] = sorted(choices) if isinstance(choices, list) else []
    return out


def _try(fn: Any) -> Any:
    try:
        return fn()
    except Exception as exc:  # audit reports every failure instead of stopping
        return {"error": str(exc)}


# A tiny two-shot analysis so agent-check exercises the real schemas without a video.
_SAMPLE_ANALYSIS: dict[str, Any] = {
    "duration": 4.0, "fps": 24.0, "width": 576, "height": 1024, "source_asset": "sample.mp4",
    "shots": [{"shot_id": "shot_001", "start": 0.0, "end": 2.0, "motion_type": "moderate",
               "camera": "handheld"},
              {"shot_id": "shot_002", "start": 2.0, "end": 4.0, "motion_type": "gentle",
               "camera": "static"}],
}


def _test_image(width: int = 96, height: int = 160) -> bytes:
    """A small gradient PNG, only to prove the model accepts an image."""
    import struct
    import zlib

    rows = b"".join(
        b"\x00" + bytes(v for x in range(width)
                         for v in (x * 255 // width, y * 255 // height, 128))
        for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2,
                                                                 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def run_agent_check(settings: Settings, *, theme: str,
                    transport: httpx.BaseTransport | None = None) -> int:
    """Prove the configured Ollama models answer every agent role with valid JSON."""
    from rokkur_studio.agents.providers import AgentOutputError, OllamaProvider
    from rokkur_studio.agents.roles import ChannelManager, CreativeDirector, DirectorOfPhotography
    from rokkur_studio.director.assets import TrackerError, load_tracker, tracker_path
    from rokkur_studio.director.prompts import Weights, build_schedule, compile_brief
    from rokkur_studio.pipeline.context import free_idle_comfyui

    hook = free_idle_comfyui(settings) if (settings.gpu.free_comfyui_before_agents
                                           and transport is None) else None
    provider = OllamaProvider(settings.ollama.url, settings.ollama.model,
                              max_retries=settings.agents.max_output_retries, transport=transport,
                              before_generate=hook)
    problem = provider.check()
    if problem:
        print(f"[FAIL] {problem}")
        return 1
    print(f"Ollama reachable at {settings.ollama.url}; model {settings.ollama.model} installed",
          flush=True)
    if settings.agents.provider != "ollama":
        print(f"note: agents.provider is '{settings.agents.provider}'; the pipeline will not use "
              "Ollama until config/studio.yaml says agents.provider: ollama")
    ok = True
    creative_input = {"theme": theme, "prompt": theme}
    print("asking creative_director (the first answer loads the model into VRAM; this can "
          "take a few minutes, longer if it runs on CPU because the GPU is full) ...",
          flush=True)
    t0 = time.monotonic()
    try:
        brief, by = CreativeDirector(provider).run(creative_input, _SAMPLE_ANALYSIS,
                                                   "youtube_short")
    except AgentOutputError as exc:
        print(f"[FAIL] creative_director: {exc}")
        return 1
    dt = time.monotonic() - t0
    if by != "ollama":
        print(f"[FAIL] creative_director fell back to rules after {dt:.1f}s (see log)")
        ok = False
    else:
        print(f"[ok] creative_director answered in {dt:.1f}s")
        print(f"     style: {brief.style}")
        for shot in brief.shot_plan:
            print(f"     {shot.shot_id} {shot.start:.1f}-{shot.end:.1f}s: {shot.intent[:90]}")
            if shot.subject:
                print(f"       subject: {shot.subject[:110]}")

    d = settings.director
    dp = provider
    if d.vision_model and d.vision_model != settings.ollama.model:
        dp = OllamaProvider(settings.ollama.url, d.vision_model,
                            max_retries=settings.agents.max_output_retries, transport=transport,
                            before_generate=hook)
        problem = dp.check()
        if problem:
            print(f"[FAIL] director.vision_model: {problem}")
            return 1
    sees = d.vision == "auto" and dp.supports_images()
    print(f"asking director_of_photography ({dp.model}; "
          f"{'can read images' if sees else 'text only'}) ...", flush=True)
    t0 = time.monotonic()
    images = {s.shot_id: _test_image() for s in brief.shot_plan} if sees else None
    framed, framing_by = DirectorOfPhotography(dp, vision=sees).run(brief, images)
    dt = time.monotonic() - t0
    if framing_by == "rule_based" or framing_by == "mixed":
        print(f"[FAIL] director_of_photography fell back to rules after {dt:.1f}s (see log)")
        ok = False
    else:
        print(f"[ok] director_of_photography answered in {dt:.1f}s ({framing_by})")
    for shot in framed.shot_plan:
        print(f"     {shot.shot_id}: {shot.shot_size} · {shot.camera_angle} · "
              f"{shot.camera_movement} · {shot.lighting}")
    try:
        tracker = load_tracker(tracker_path(settings.studio.data_dir))
    except TrackerError as exc:
        print(f"[FAIL] asset tracker: {exc}")
        return 1
    compiled = compile_brief(framed, creative_input=creative_input, tracker=tracker, anchor=None,
                             character_key=None,
                             weights=Weights(framing=d.framing_weight, angle=d.angle_weight))
    print(f"     prompt: {compiled.shot_plan[0].prompt[:300]}")
    schedule = build_schedule(compiled.shot_plan, compiled.negative_prompt,
                              duration=_SAMPLE_ANALYSIS["duration"], fps=d.schedule_fps,
                              interval=d.schedule_interval,
                              inline_negative=d.schedule_inline_negative)
    print(f"     schedule: {len(schedule.keyframes)} keyframes, max_frames {schedule.max_frames}")
    print("asking channel_manager ...", flush=True)
    t0 = time.monotonic()
    draft, by = ChannelManager(provider).run(creative_input, brief.model_dump(), "youtube_short",
                                             _SAMPLE_ANALYSIS["duration"])
    dt = time.monotonic() - t0
    if draft is None:
        print(f"[FAIL] channel_manager fell back to rules after {dt:.1f}s (see log)")
        ok = False
    else:
        print(f"[ok] channel_manager answered in {dt:.1f}s")
        print(f"     title: {draft.title}")
        print(f"     tags: {', '.join(draft.tags)}")
    return 0 if ok else 1


def cmd_agent_check(args: argparse.Namespace) -> int:
    return run_agent_check(_settings(args), theme=args.theme)


def cmd_audit(args: argparse.Namespace) -> int:
    """Inspect the local environment and write data/audit.json (Phase 0 on the workstation)."""
    from sqlalchemy import create_engine, text

    settings = _settings(args)
    report: dict[str, Any] = {"platform": platform.platform(), "python": sys.version}

    def db() -> Any:
        engine = create_engine(settings.database.url)
        with engine.connect() as c:
            return {"ok": True, "version": c.execute(text("select version()")).scalar()}

    report["database"] = _try(db)
    report["ffmpeg"] = {"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}
    report["docker"] = _try(lambda: subprocess.run(
        ["docker", "info", "--format", "{{json .}}"], capture_output=True, text=True,
        timeout=20).stdout[:2000] or "docker not available")
    report["comfyui_system_stats"] = _try(
        lambda: httpx.get(f"{settings.comfyui.url}/system_stats", timeout=5).json())
    object_info = _try(lambda: httpx.get(f"{settings.comfyui.url}/object_info", timeout=30).json())
    report["comfyui_node_classes"] = (object_info if "error" in object_info else sorted(object_info))
    report["comfyui_models"] = (object_info if "error" in object_info else comfy_model_files(object_info))
    report["ollama_models"] = _try(
        lambda: httpx.get(f"{settings.ollama.url}/api/tags", timeout=5).json())
    report["ollama_loaded"] = _try(
        lambda: httpx.get(f"{settings.ollama.url}/api/ps", timeout=5).json())
    out = Path(settings.studio.data_dir) / "audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    summary = {k: ("error" not in v) if isinstance(v, dict) else bool(v)
               for k, v in report.items() if k not in ("platform", "python")}
    print(json.dumps(summary, indent=2))
    print(f"full report: {out}")
    return 0


def cmd_youtube_auth(args: argparse.Namespace) -> int:
    """Sign in to YouTube once; the refresh token lands in secrets/ (git-ignored)."""
    from rokkur_studio.youtube.client import YouTubeClient, YouTubeError
    from rokkur_studio.youtube.oauth import OAuthClient, OAuthError, TokenStore, installed_flow

    settings = _settings(args)
    yt = settings.youtube
    try:
        client = OAuthClient.load(yt.client_secret_path)
    except OAuthError as exc:
        print(f"[FAIL] {exc}")
        return 1
    store = TokenStore(yt.token_path)
    if args.status:
        if not store.exists():
            print(f"not signed in ({yt.token_path} missing)")
            return 1
    elif args.sign_out:
        store.delete()
        print("signed out: token deleted (revoke the app at myaccount.google.com/permissions "
              "if you want Google to forget it too)")
        return 0
    else:
        def show(url: str) -> None:
            print("Open this link in your browser and allow access:\n\n  " + url + "\n")
            if args.paste:
                print("After allowing, the browser lands on a 127.0.0.1 page that may not "
                      "load. Copy its full address from the address bar and paste it here.")
            else:
                print(f"Waiting for Google to send the browser back to port {yt.auth_port} …")

        try:
            with httpx.Client(timeout=30) as http:
                installed_flow(client, store, port=yt.auth_port, http=http, open_url=show,
                               paste=args.paste, read_line=lambda: input("redirect URL: "))
        except OAuthError as exc:
            print(f"[FAIL] {exc}")
            return 1
        except OSError as exc:
            print(f"[FAIL] could not listen on port {yt.auth_port}: {exc}. Run again with "
                  "--paste, or publish the port (studio.ps1 youtube-auth does).")
            return 1
        print(f"signed in; token saved to {yt.token_path}")
    api = YouTubeClient(client, store)
    try:
        ch = api.my_channel()
    except (OAuthError, YouTubeError) as exc:
        print(f"[FAIL] token does not work: {exc}")
        return 1
    finally:
        api.close()
    print(f"[ok] signed in as channel '{ch['title']}' ({ch.get('custom_url') or ch['id']})")
    if not yt.enabled:
        print("note: youtube.enabled is false; set STUDIO_YOUTUBE__ENABLED=true in .env "
              "before `publish` will upload anything")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    """Upload one finished project to YouTube (private unless told otherwise)."""
    from rokkur_studio.pipeline.context import build_context
    from rokkur_studio.services import publishing
    from rokkur_studio.services.projects import get_project, latest_document
    from rokkur_studio.youtube.client import YouTubeError, video_url
    from rokkur_studio.youtube.oauth import OAuthError

    settings = _settings(args)
    ctx = build_context(settings)
    with ctx.db.session() as s:
        project = get_project(s, args.project_id)
        meta = latest_document(s, project.id, "metadata")
        print(f"project {project.id} '{project.name}' is {project.status}")
        if meta:
            print(f"  title: {meta.data.get('title')}")
            print(f"  tags: {', '.join(meta.data.get('tags') or [])}")
            for w in meta.data.get("warnings") or []:
                print(f"  warning: {w}")
        try:
            privacy = publishing.resolve_privacy(settings, project,
                                                 args.privacy or ("private" if args.at else None))
            when = None
            if args.at == "next":
                when = publishing.next_release_slot(s, settings, for_project=project.id)
                if when is None:
                    print("[FAIL] no free release time: set youtube.release_times and "
                          "youtube.allow_public in config/studio.yaml")
                    return 1
            elif args.at:
                when = publishing.parse_when(args.at, settings)
            if when is not None:
                when = publishing.resolve_schedule(settings, privacy, when)
            playlist = publishing.find_playlist(settings, args.playlist) if args.playlist else (
                settings.youtube.default_playlist_id or None)
        except publishing.PublishGateError as exc:
            print(f"[FAIL] {exc}")
            return 1
    plan = privacy + (f", goes public {publishing.local_label(settings, when)}" if when else "")
    print(f"  privacy: {plan}")
    if playlist:
        print(f"  playlist: {publishing.playlist_title(settings, playlist) or playlist}")
    if args.dry_run:
        with ctx.db.transaction() as s:
            project = get_project(s, args.project_id, for_update=True)
            try:
                pub = publishing.dry_run(s, project, privacy=privacy, publish_at=when,
                                         playlist_id=playlist, actor="cli-publish",
                                         settings=settings)
            except publishing.PublishGateError as exc:
                print(f"[FAIL] {exc}")
                return 1
            print("dry run OK; this is the request that would be sent:")
            print(json.dumps(pub.request["body"], indent=2))
        return 0
    if not args.yes:
        answer = input(f"Upload to YouTube ({plan})? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("cancelled")
            return 1
    failure: YouTubeError | None = None
    try:
        with publishing.youtube_client(settings) as client, ctx.db.transaction() as s:
            project = get_project(s, args.project_id, for_update=True)
            try:
                pub = publishing.upload(s, project, settings=settings, store=ctx.store,
                                        client=client, privacy=privacy, publish_at=when,
                                        playlist_id=playlist, actor="cli-publish")
            except YouTubeError as exc:
                failure = exc  # caught inside the transaction so the failed attempt is kept
            else:
                video_id = pub.youtube_video_id or ""
                warning = pub.error
    except (publishing.PublishGateError, OAuthError) as exc:
        print(f"[FAIL] {exc}")
        return 1
    if failure is not None:
        print(f"[FAIL] upload failed: {failure}")
        return 1
    print(f"[ok] uploaded ({plan}): {video_url(video_id)}")
    if warning:
        print(f"  warning: {warning.get('message')}")
    return 0


def cmd_youtube_playlists(args: argparse.Namespace) -> int:
    """Load the signed-in channel's playlists so publish/the dashboard can offer them."""
    from rokkur_studio.services import publishing
    from rokkur_studio.youtube.client import YouTubeError
    from rokkur_studio.youtube.oauth import OAuthError

    settings = _settings(args)
    try:
        with publishing.youtube_client(settings, uploads=False) as client:
            data = publishing.refresh_playlists(settings, client)
    except (OAuthError, YouTubeError) as exc:
        print(f"[FAIL] {exc}")
        return 1
    for pl in data["items"]:
        default = "  (default)" if pl["id"] == settings.youtube.default_playlist_id else ""
        print(f"{pl['id']}  {pl['title']}  [{pl.get('privacy') or '?'}, "
              f"{pl.get('videos') if pl.get('videos') is not None else '?'} videos]{default}")
    if not data["items"]:
        print("this channel has no playlists")
    print(f"saved to {publishing.playlists_path(settings)}")
    return 0


def cmd_smoke_test(args: argparse.Namespace) -> int:
    """End-to-end fixture: synthetic legal video → analysis → render → QC → encode → dry-run."""
    from rokkur_studio.api.schemas import CreativeIn, ProjectCreate, RightsIn, SourceIn
    from rokkur_studio.domain.rights import RightsCategory
    from rokkur_studio.pipeline.context import build_context
    from rokkur_studio.services import commands

    settings = _settings(args)
    if args.renderer:
        settings.render.renderer = args.renderer
    ctx = build_context(settings)
    fixtures = Path(settings.studio.data_dir) / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    video = fixtures / "smoke_source.mp4"
    if not video.exists():
        ctx.ffmpeg.make_test_video(video, seconds=4)
    creative: dict[str, Any] = {"theme": "1970s stop-motion sci-fi",
                                "prompt": "clay robot walking through a retro space station"}
    if args.inject_fault:
        creative["test_faults"] = {"shot_002": {"kind": "black", "attempts": [1]}}
    with ctx.db.transaction() as s:
        project = commands.create_project(s, ProjectCreate(
            name="Smoke test", source=SourceIn(platform="local", local_path=str(video.resolve())),
            rights=RightsIn(category=RightsCategory.USER_OWNED,
                            permission_evidence="synthetic test pattern generated by FFmpeg"),
            creative=CreativeIn(**creative), render_profile=args.profile, autostart=True),
            settings, actor="smoke-test")
        pid = project.id
    return _drive(ctx, pid, args.timeout, actor="smoke-test")


def _drive(ctx: Any, pid: str, timeout: float, *, actor: str) -> int:
    """Run the pipeline for one project in-process, then print its trail and the dry-run upload."""
    from rokkur_studio.db.models import Event
    from rokkur_studio.jobs.worker import Worker
    from rokkur_studio.services import publishing
    from rokkur_studio.services.projects import get_project, latest_document

    worker = Worker(ctx, worker_id=actor)
    deadline = time.monotonic() + timeout
    done = {"READY_TO_PUBLISH", "FAILED", "RIGHTS_REJECTED", "CANCELLED", "RIGHTS_PENDING"}
    while time.monotonic() < deadline:
        worker.drain()
        with ctx.db.session() as s:
            status = get_project(s, pid).status
        if status in done:
            break
        time.sleep(0.5)
    with ctx.db.transaction() as s:
        project = get_project(s, pid)
        print(f"project {pid}: {project.status}")
        for e in s.query(Event).filter_by(project_id=pid).order_by(Event.id):
            if e.to_state:
                print(f"  {e.from_state or '':<24} -> {e.to_state:<24} ({e.actor})")
        if project.status != "READY_TO_PUBLISH":
            return 1
        qc = latest_document(s, pid, "qc_report")
        print(f"QC: {qc.data['decision']} overall={qc.data['overall']}" if qc else "QC: none")
        pub = publishing.dry_run(s, project, actor=actor)
        print("dry-run upload request:")
        print(json.dumps(pub.request["body"], indent=2))
        print(f"final video: {ctx.store.path_for(f'{pid}/final/final.mp4')}")
        print(f"  (on Windows: data\\projects\\{pid}\\final\\final.mp4 in the studio folder)")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render a video you have rights to through the full pipeline (publishing stays a dry run)."""
    from rokkur_studio.api.schemas import CreativeIn, ProjectCreate, RightsIn, SourceIn
    from rokkur_studio.domain.rights import RightsCategory
    from rokkur_studio.pipeline.context import build_context, profile_availability
    from rokkur_studio.services import commands

    settings = _settings(args)
    settings.render.renderer = args.renderer
    source = Path(args.path)
    if not source.is_file():
        print(f"no such file: {source} (inside Docker your media folder is /media)", file=sys.stderr)
        return 2
    ctx = build_context(settings)
    profile = args.profile or settings.render.default_profile
    try:
        settings.profile(profile)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if problem := profile_availability(ctx)[profile]:
        print(problem, file=sys.stderr)
        return 2
    with ctx.db.transaction() as s:
        project = commands.create_project(s, ProjectCreate(
            name=args.name or source.stem, target_format=args.format, render_profile=args.profile,
            source=SourceIn(platform="local", local_path=str(source.resolve())),
            rights=RightsIn(category=RightsCategory(args.rights), permission_evidence=args.evidence),
            creative=CreativeIn(theme=args.theme, prompt=args.prompt, subject=args.subject,
                                reference_mode=args.reference, character_key=args.character,
                                seed=args.seed,
                                canny_low=args.canny[0] if args.canny else None,
                                canny_high=args.canny[1] if args.canny else None,
                                use_global_look=not args.no_global_look), autostart=True),
            settings, actor="cli")
        pid = project.id
    print(f"project {pid} created; rendering with {args.renderer} / {args.profile or 'default profile'}")
    return _drive(ctx, pid, args.timeout, actor="cli-render")


def cmd_taste(args: argparse.Namespace) -> int:
    """Print what your ratings say works: plain text to paste to a collaborator."""
    from rokkur_studio.db.session import Database
    from rokkur_studio.services import taste

    settings = _settings(args)
    with Database(settings.database.url).session() as s:
        print(taste.report(taste.build_profile(s)))
    return 0


def cmd_timings(args: argparse.Namespace) -> int:
    """Print where a project's time went (latest project by default), to compare settings."""
    from rokkur_studio.db.session import Database
    from rokkur_studio.services import timings

    settings = _settings(args)
    with Database(settings.database.url).session() as s:
        pid = args.project_id or timings.latest_project_id(s)
        if pid is None:
            print("no projects yet", file=sys.stderr)
            return 1
        try:
            data = timings.collect(s, pid)
        except LookupError:
            print(f"no project {pid}", file=sys.stderr)
            return 2
    print(json.dumps(data, indent=2, default=str) if args.json else timings.report(data))
    return 0


def cmd_prompt_schedule(args: argparse.Namespace) -> int:
    """Print a project's Batch Prompt Schedule, ready to paste into FizzNodes."""
    from rokkur_studio.db.session import Database
    from rokkur_studio.services.projects import get_project, latest_document

    settings = _settings(args)
    with Database(settings.database.url).session() as s:
        try:
            get_project(s, args.project_id)
        except LookupError:
            print(f"no project {args.project_id}", file=sys.stderr)
            return 2
        doc = latest_document(s, args.project_id, "prompt_schedule")
    if doc is None:
        print("this project has no prompt schedule yet: the brief stage writes it",
              file=sys.stderr)
        return 1
    print(doc.data["text"])
    print(f"\n(max_frames {doc.data['max_frames']}, {doc.data['fps']} fps)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    from rokkur_studio.domain.rights import RightsCategory

    parser = argparse.ArgumentParser(prog="rokkur-studio")
    parser.add_argument("--config", help="config directory (default: ./config)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("migrate", help="apply database migrations")
    p.add_argument("revision", nargs="?", default="head")
    p.set_defaults(func=cmd_migrate)
    p = sub.add_parser("api", help="run the Studio API + dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8400)
    p.set_defaults(func=cmd_api)
    p = sub.add_parser("worker", help="run a job worker")
    p.add_argument("--kinds", help="comma-separated job kinds to accept")
    p.set_defaults(func=cmd_worker)
    p = sub.add_parser("comfy-check", help="validate ComfyUI + templates")
    p.add_argument("--cloud", action="store_true",
                   help="check the cloud ComfyUI server (STUDIO_CLOUD__URL) instead of this PC's")
    p.set_defaults(func=cmd_comfy_check)
    sub.add_parser("audit", help="inspect the local environment").set_defaults(func=cmd_audit)
    p = sub.add_parser("agent-check",
                       help="ask the Ollama agents for a brief, framing and metadata")
    p.add_argument("--theme", default="1970s stop-motion claymation, warm film grain")
    p.set_defaults(func=cmd_agent_check)
    p = sub.add_parser("youtube-auth", help="sign in to YouTube (token stays in secrets/)")
    p.add_argument("--paste", action="store_true",
                   help="paste the redirect URL instead of listening on the auth port")
    p.add_argument("--status", action="store_true", help="only check the saved sign-in")
    p.add_argument("--sign-out", action="store_true", help="delete the saved token")
    p.set_defaults(func=cmd_youtube_auth)
    p = sub.add_parser("publish", help="upload a READY_TO_PUBLISH project to YouTube")
    p.add_argument("project_id")
    p.add_argument("--privacy", choices=["private", "unlisted", "public"],
                   help="default: private")
    p.add_argument("--at", metavar="WHEN",
                   help="make it public at this time, e.g. '2026-10-09 18:00' (youtube.timezone)"
                        " or 'next' for the next free release time; uploads as private until then")
    p.add_argument("--playlist", help="playlist id or exact title (see youtube-playlists)")
    p.add_argument("--dry-run", action="store_true", help="show the request, upload nothing")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_publish)
    sub.add_parser("youtube-playlists", help="load the channel's playlists").set_defaults(
        func=cmd_youtube_playlists)
    p = sub.add_parser("smoke-test", help="run the end-to-end fixture")
    p.add_argument("--renderer", choices=["ffmpeg_preview", "comfyui"])
    p.add_argument("--profile", default="PREVIEW")
    p.add_argument("--inject-fault", action="store_true", help="exercise QC failure + repair")
    p.add_argument("--timeout", type=float, default=300)
    p.set_defaults(func=cmd_smoke_test)
    p = sub.add_parser("render", help="render a video you have rights to")
    p.add_argument("path", help="video file, e.g. /media/clip.mp4")
    p.add_argument("--theme", required=True, help="look to apply, e.g. '1970s claymation'")
    p.add_argument("--prompt", help="extra detail for the style prompt")
    p.add_argument("--character", help="character name from the Director page, e.g. NEO")
    p.add_argument("--subject", choices=["auto", "keep", "restyle"], default="auto",
                   help="keep the real main subject over the render, restyle it, or let the "
                   "studio decide from the prompt (default)")
    p.add_argument("--reference", choices=["auto", "cutout", "source", "none"], default="auto",
                   help="image Wan gets as a reference: a cutout of the real subject when it is "
                   "kept (auto, default), always a cutout, the first source frame, or none")
    p.add_argument("--canny", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   help="edge thresholds for the Canny workflows (default 0.2 0.5)")
    p.add_argument("--seed", type=int, help="one seed for every shot, to compare settings")
    p.add_argument("--no-global-look", action="store_true",
                   help="skip the global prefix, style modifiers and negative prompt")
    p.add_argument("--rights", required=True,
                   choices=[c.value for c in RightsCategory if c.value not in ("UNKNOWN", "REJECTED")],
                   help="why you may use this video; unknown rights are never rendered")
    p.add_argument("--evidence", required=True, help="note backing the rights claim")
    p.add_argument("--profile", help="render profile (default from config)")
    p.add_argument("--renderer", choices=["ffmpeg_preview", "comfyui"], default="comfyui")
    p.add_argument("--format", choices=["youtube_short", "youtube_video"], default="youtube_short")
    p.add_argument("--name")
    p.add_argument("--timeout", type=float, default=4 * 3600)
    p.set_defaults(func=cmd_render)
    p = sub.add_parser("prompt-schedule", help="print a project's Batch Prompt Schedule")
    p.add_argument("project_id")
    p.set_defaults(func=cmd_prompt_schedule)
    sub.add_parser("taste", help="print what your ratings say works").set_defaults(
        func=cmd_taste)
    p = sub.add_parser("timings", help="print where a project's time went (default: latest)")
    p.add_argument("project_id", nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_timings)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
