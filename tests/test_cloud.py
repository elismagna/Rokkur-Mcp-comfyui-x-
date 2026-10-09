"""Local and cloud rendering: a video can render on a ComfyUI server elsewhere (docs/cloud.md)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select

from rokkur_studio.api.schemas import CreativeIn, ProjectCreate, RightsIn, SourceIn
from rokkur_studio.comfyui.client import ComfyClient
from rokkur_studio.config import load_settings
from rokkur_studio.db.models import CostEntry, GpuLease, Project
from rokkur_studio.domain.rights import RightsCategory
from rokkur_studio.services import commands
from rokkur_studio.services.projects import budget_usage, failure_reason
from tests.fakes import FakeComfyUI
from tests.test_dashboard import client_for
from tests.test_pipeline import create, run, status

ROOT = Path(__file__).resolve().parents[1]
CLOUD_URL = "http://127.0.0.1:9"  # nothing listens here: the services probe fails fast


def with_cloud(ctx, *, price: float = 0.6) -> tuple[FakeComfyUI, FakeComfyUI, list[str]]:
    """This PC's ComfyUI and a cloud one, both fakes; returns them and the cloud's auth headers."""
    ctx.settings.render.renderer = "comfyui"
    ctx.settings.comfyui.poll_interval_s = 0
    cloud = ctx.settings.cloud
    cloud.enabled, cloud.url, cloud.price_per_hour_usd = True, CLOUD_URL, price
    cloud.token = SecretStr("s3cret")
    local, remote, auth = FakeComfyUI(), FakeComfyUI(), []

    def handle(request: httpx.Request) -> httpx.Response:
        auth.append(request.headers.get("authorization", ""))
        return remote.handle(request)

    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=local.transport())
    ctx.cloud_factory = lambda: ComfyClient(cloud.url, transport=httpx.MockTransport(handle),
                                            headers=cloud.headers())
    return local, remote, auth


def test_cloud_settings_come_from_env_and_the_token_stays_secret():
    s = load_settings(ROOT / "config", environ={
        "STUDIO_CLOUD__ENABLED": "true", "STUDIO_CLOUD__URL": "https://gpu.example",
        "STUDIO_CLOUD__TOKEN": "1234", "STUDIO_CLOUD__AUTH_SCHEME": ""})
    assert s.cloud.ready and s.cloud.headers() == {"Authorization": "1234"}
    assert "1234" not in repr(s.cloud) and "1234" not in str(s.cloud.model_dump())
    assert not load_settings(ROOT / "config", environ={}).cloud.ready  # off by default


def test_profiles_are_checked_against_the_server_they_render_on(settings):
    settings.render.renderer = "comfyui"
    assert "not set up" in settings.profile_problem("RTX3070_QUALITY", target="cloud")
    settings.cloud.enabled, settings.cloud.url = True, "https://gpu.example"
    assert settings.profile_problem("RTX3070_QUALITY", target="cloud") is None
    # a 24 GB profile cannot run on this 8 GB PC but can on a 24 GB cloud GPU
    assert "Needs 24 GB" in settings.profile_problem("FUTURE_24GB")
    assert settings.profile_problem("FUTURE_24GB", target="cloud") is None
    assert "choose Cloud" in settings.profile_problem("HYBRID_MAX")
    settings.cloud.default = "cloud"
    assert settings.new_project_target(None) == "cloud"
    assert settings.new_project_target("local") == "local"


def test_a_cloud_video_is_refused_when_cloud_is_not_set_up(ctx, sample_video):
    ctx.settings.render.renderer = "comfyui"
    with ctx.db.transaction() as s, pytest.raises(ValueError, match="not set up"):
        commands.create_project(s, ProjectCreate(
            name="q", source=SourceIn(local_path=str(sample_video)),
            rights=RightsIn(category=RightsCategory.USER_OWNED, permission_evidence="mine"),
            creative=CreativeIn(theme="x", render_on="cloud")), ctx.settings)


def test_a_cloud_video_renders_on_the_cloud_server_without_the_local_gpu(ctx, sample_video):
    local, remote, auth = with_cloud(ctx)
    hooks: list[str] = []
    ctx.gpu.before_heavy = [hooks.append]  # would unload Ollama on this PC
    ctx.gpu.after_heavy = [hooks.append]   # would free this PC's ComfyUI
    pid = create(ctx, sample_video, render_on="cloud")
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    assert len(remote.prompts) == 2 and not local.prompts  # one prompt per shot, all remote
    assert hooks == []
    assert auth and set(auth) == {"Bearer s3cret"}
    with ctx.db.session() as s:
        assert s.scalars(select(GpuLease)).all() == []  # this PC's GPU was never locked
        costs = s.scalars(select(CostEntry).where(CostEntry.project_id == pid)).all()
        assert {c.kind for c in costs} == {"cloud_gpu_minutes"}
        assert sum(c.usd for c in costs) > 0
        used = budget_usage(s, pid, ctx.settings)
        assert used["cloud"] > 0 and used["gpu"] == 0 and not used["exhausted"]
    page = client_for(ctx).get(f"/ui/projects/{pid}").text
    assert "cloud GPU minutes" in page and "estimated cost" in page


def test_a_local_video_still_renders_here_with_cloud_set_up(ctx, sample_video):
    local, remote, _ = with_cloud(ctx)
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    assert len(local.prompts) == 2 and not remote.prompts
    with ctx.db.session() as s:
        assert s.get(Project, pid).creative_input["render_on"] == "local"


def test_the_cloud_minute_budget_stops_cloud_videos_only(ctx, sample_video):
    _, remote, _ = with_cloud(ctx)
    ctx.settings.costs.max_cloud_gpu_minutes = 0  # no cloud minutes allowed
    local_pid = create(ctx, sample_video, autostart=False)
    pid = create(ctx, sample_video, render_on="cloud")
    run(ctx)
    assert status(ctx, pid) == "FAILED" and not remote.prompts
    with ctx.db.session() as s:
        assert "cloud GPU budget" in failure_reason(s, s.get(Project, pid))
        assert not budget_usage(s, local_pid, ctx.settings)["exhausted"]


def test_the_new_video_page_offers_cloud_only_when_it_is_set_up(ctx):
    ctx.settings.render.renderer = "comfyui"
    c = client_for(ctx)
    assert "Where to render" not in c.get("/ui/new").text
    with_cloud(ctx)
    c = client_for(ctx)
    page = c.get("/ui/new").text
    assert "Where to render" in page and "about $0.6 an hour" in page
    system = c.get("/ui/system").text
    assert "Cloud ComfyUI" in system and "s3cret" not in system and "127.0.0.1" in system
