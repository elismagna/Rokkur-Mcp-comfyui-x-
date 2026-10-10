"""Starting and stopping the RunPod cloud GPU (docs/cloud.md, "Start the cloud GPU with the studio")."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from rokkur_studio import cli
from rokkur_studio.config import load_settings
from rokkur_studio.db.models import Job
from rokkur_studio.pipeline import context as context_mod
from rokkur_studio.services.runpod import (
    IdleStopper,
    PodStatusCache,
    RunPodClient,
    RunPodError,
    connect_info,
    parse_pod,
)
from tests.test_dashboard import client_for

ROOT = Path(__file__).resolve().parents[1]
KEY = "rpa_TESTKEY123"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


class FakeRunPod:
    """RunPod's REST API for one pod: GET /pods/{id}, POST /pods/{id}/start and /stop."""

    def __init__(self, status: str = "EXITED", *, boot_polls: int = 0) -> None:
        self.status, self.boot_polls = status, boot_polls
        self.calls: list[str] = []
        self.auth: set[str] = set()
        self.started_at = "2026-10-10T09:00:00.000Z"

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.auth.add(request.headers.get("authorization", ""))
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": "unauthorized"})
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        if not path.startswith("/v1/pods/pod123"):
            return httpx.Response(404, json={"error": "Pod not found"})
        if path.endswith("/start"):
            self.status = "RUNNING"
            return httpx.Response(200)
        if path.endswith("/stop"):
            self.status = "EXITED"
            return httpx.Response(200)
        booting = self.status == "RUNNING" and self.boot_polls > 0
        if booting:
            self.boot_polls -= 1
        up = self.status == "RUNNING" and not booting
        return httpx.Response(200, json={
            "id": "pod123", "name": "rokkur-4090", "desiredStatus": self.status,
            "publicIp": "69.145.85.70" if up else "", "portMappings": {"22": 10705} if up else {},
            "costPerHr": "0.69", "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1,
                                         "displayName": "RTX 4090"},
            "lastStartedAt": self.started_at})

    def client(self) -> RunPodClient:
        return RunPodClient(KEY, "pod123", transport=httpx.MockTransport(self.handle))


def with_pod(ctx, fake: FakeRunPod) -> FakeRunPod:
    cloud = ctx.settings.cloud
    cloud.enabled, cloud.url = True, "http://host.docker.internal:8189"
    cloud.runpod_api_key, cloud.runpod_pod_id = SecretStr(KEY), "pod123"
    ctx.runpod_factory = fake.client
    return fake


def test_settings_keep_the_key_secret_and_ids_as_text():
    s = load_settings(ROOT / "config", environ={
        "STUDIO_CLOUD__ENABLED": "true", "STUDIO_CLOUD__URL": "http://host.docker.internal:8189",
        "STUDIO_CLOUD__RUNPOD_API_KEY": "123456", "STUDIO_CLOUD__RUNPOD_POD_ID": "1e5",
        "STUDIO_CLOUD__SSH_KEY": r"C:\Users\Elis\.ssh\id_ed25519"})
    cloud = s.cloud
    assert cloud.pod_control and cloud.runpod_pod_id == "1e5"  # not the number 100000.0
    assert cloud.runpod_api_key.get_secret_value() == "123456"
    assert "123456" not in repr(cloud) and "123456" not in str(cloud.model_dump())
    assert cloud.ssh_key == r"C:\Users\Elis\.ssh\id_ed25519" and cloud.local_port == 8189
    assert cloud.auto_stop_idle_minutes == 30 and cloud.ask_on_launch
    plain = load_settings(ROOT / "config", environ={}).cloud
    assert not plain.pod_control and plain.local_port == 8189


def test_the_client_reads_the_pod_and_starts_and_stops_it():
    fake = FakeRunPod("RUNNING")
    pod = fake.client().state()
    assert pod.reachable and (pod.ip, pod.ssh_port) == ("69.145.85.70", 10705)
    assert pod.price_per_hour == 0.69 and pod.gpu == "RTX 4090"
    assert pod.label() == "Cloud GPU on · RTX 4090 · $0.69/h"
    assert pod.started_at == datetime(2026, 10, 10, 9, 0, tzinfo=UTC)
    client = fake.client()
    client.stop()
    client.start()
    assert fake.calls[-2:] == ["POST /v1/pods/pod123/stop", "POST /v1/pods/pod123/start"]
    assert fake.auth == {f"Bearer {KEY}"}
    assert not parse_pod({"desiredStatus": "RUNNING", "portMappings": {}}).reachable


def test_runpod_errors_explain_without_leaking_the_key():
    fake = FakeRunPod()
    with pytest.raises(RunPodError, match="refused the API key") as err:
        RunPodClient("wrong-key", "pod123", transport=httpx.MockTransport(fake.handle)).state()
    assert "wrong-key" not in str(err.value)
    with pytest.raises(RunPodError, match="no pod 'other'"):
        RunPodClient(KEY, "other", transport=httpx.MockTransport(fake.handle)).state()


def test_waiting_for_the_pod_until_its_ssh_port_is_known():
    fake = FakeRunPod("RUNNING", boot_polls=2)
    slept: list[float] = []
    pod = fake.client().wait_until_reachable(poll_s=5, sleep=slept.append, clock=lambda: 0.0)
    assert pod.reachable and slept == [5, 5]
    t = iter(range(0, 1000, 100))
    with pytest.raises(RunPodError, match="did not come up"):
        FakeRunPod("RUNNING", boot_polls=99).client().wait_until_reachable(
            timeout_s=300, sleep=lambda _: None, clock=lambda: float(next(t)))


def test_connect_info_gives_the_launcher_a_safe_remote_command(settings):
    settings.cloud.url = "http://host.docker.internal:8190"
    info = connect_info(settings.cloud, FakeRunPod("RUNNING").client().state())
    assert (info["ip"], info["ssh_port"], info["local_port"]) == ("69.145.85.70", 10705, 8190)
    cmd = info["remote_start"]
    assert '"' not in cmd  # Windows PowerShell 5.1 mangles double quotes passed to ssh.exe
    assert "127.0.0.1:8188/system_stats ||" in cmd and "cd /workspace/ComfyUI" in cmd
    assert "--listen 127.0.0.1 --port 8188 > /workspace/comfyui.log" in cmd
    assert info["ssh_user"] == "root" and info["ask_on_launch"] is True


def test_the_cloud_pod_command_prints_one_json_line_for_the_launcher(ctx, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_settings", lambda args: ctx.settings)
    monkeypatch.setattr(context_mod, "build_context", lambda settings, db=None: ctx)

    def last_json() -> dict:
        lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith('{"configured"')]
        return json.loads(lines[-1])

    assert cli.main(["cloud-pod", "status", "--json"]) == 0
    assert last_json() == {"configured": False, "ready": False}
    assert cli.main(["cloud-pod", "start", "--json"]) == 1
    fake = with_pod(ctx, FakeRunPod("EXITED", boot_polls=1))
    monkeypatch.setattr("rokkur_studio.services.runpod.time.sleep", lambda _: None)
    assert cli.main(["cloud-pod", "start", "--wait", "--json"]) == 0
    out = last_json()
    assert out["started"] and out["running"] and out["ssh_port"] == 10705
    assert out["local_port"] == 8189 and "remote_start" in out and KEY not in json.dumps(out)
    assert fake.calls.count("POST /v1/pods/pod123/start") == 1
    assert cli.main(["cloud-pod", "start", "--json"]) == 0  # already on: no second start
    assert not last_json()["started"] and fake.calls.count("POST /v1/pods/pod123/start") == 1
    assert cli.main(["cloud-pod", "stop", "--json"]) == 0 and last_json()["stopped"]
    assert cli.main(["cloud-pod", "status"]) == 0
    assert "Cloud GPU stopped" in capsys.readouterr().out
    ctx.settings.cloud.runpod_api_key = SecretStr("wrong")
    ctx.runpod_factory = lambda: RunPodClient("wrong", "pod123",
                                              transport=httpx.MockTransport(fake.handle))
    assert cli.main(["cloud-pod", "status", "--json"]) == 1
    assert "refused the API key" in last_json()["error"]


def test_the_dashboard_shows_the_pod_and_can_stop_it(ctx):
    fake = with_pod(ctx, FakeRunPod("RUNNING"))
    c = client_for(ctx)
    page = c.get("/ui/system").text
    assert 'id="cloud-gpu"' in page and "RTX 4090" in page and "$0.69 an hour" in page
    assert "Stop the cloud GPU" in page and KEY not in page
    assert c.get("/ui/status").json()["cloud_gpu"] == {
        "on": True, "label": "Cloud GPU on · RTX 4090 · $0.69/h"}
    with ctx.db.transaction() as s:  # a picture rendering on the cloud right now
        s.add(Job(id="job_cloudpic", kind="image", status="RUNNING", payload={"render_on": "cloud"}))
    r = c.post("/ui/cloud-gpu/stop", follow_redirects=False)
    assert r.status_code == 303 and "err=" in r.headers["location"] and fake.status == "RUNNING"
    with ctx.db.transaction() as s:
        job = s.get(Job, "job_cloudpic")
        job.status, job.finished_at = "SUCCEEDED", datetime.now(UTC)
    r = c.post("/ui/cloud-gpu/stop", follow_redirects=False)
    assert r.status_code == 303 and "msg=Stopping" in r.headers["location"]
    assert fake.status == "EXITED"
    page = c.get("/ui/system").text
    assert "Stopped" in page and "Stop the cloud GPU" not in page


def test_no_pod_control_means_no_cloud_gpu_card(ctx):
    c = client_for(ctx)
    assert 'id="cloud-gpu"' not in c.get("/ui/system").text
    assert c.get("/ui/status").json()["cloud_gpu"] is None


def test_the_status_cache_answers_at_once_and_refreshes_in_the_background():
    now = [0.0]
    fetched: list[int] = []

    def fetch():
        fetched.append(1)
        return parse_pod({"desiredStatus": "RUNNING"})

    cache = PodStatusCache(fetch, max_age_s=60, clock=lambda: now[0])
    assert cache.get(wait=True)[0].running and len(fetched) == 1
    now[0] = 30
    cache.get()
    assert len(fetched) == 1  # still fresh
    cache.invalidate()
    cache.get(wait=True)
    assert len(fetched) == 2
    broken = PodStatusCache(lambda: (_ for _ in ()).throw(RunPodError("down")))
    assert broken.get(wait=True) == (None, "down")


def _stopper(ctx, fake: FakeRunPod, *, busy=None, minutes: float = 30) -> IdleStopper:
    return IdleStopper(fake.client, minutes, ctx.db.session, busy_probe=busy,
                       check_every_s=0, now=lambda: NOW)


def test_an_idle_pod_is_stopped_but_a_busy_one_is_not(ctx):
    fake = FakeRunPod("RUNNING")  # started 3 hours before NOW
    with ctx.db.transaction() as s:
        s.add(Job(id="job_old", kind="image", status="SUCCEEDED", payload={"render_on": "cloud"},
                  finished_at=NOW - timedelta(minutes=10)))
        s.add(Job(id="job_local", kind="image", status="RUNNING", payload={"render_on": "local"}))
        s.add(Job(id="job_later", kind="image", status="QUEUED", payload={"render_on": "cloud"},
                  run_after=NOW + timedelta(hours=5)))  # scheduled for later: does not count
    assert _stopper(ctx, fake).tick() is None and fake.status == "RUNNING"  # used 10 min ago
    assert _stopper(ctx, fake, minutes=5).tick().startswith("Stopped the cloud GPU after 5 min")
    assert fake.status == "EXITED"
    fake.status = "RUNNING"
    assert _stopper(ctx, fake, minutes=5, busy=lambda: True).tick() is None  # ComfyUI in use
    with ctx.db.transaction() as s:
        s.get(Job, "job_later").run_after = NOW - timedelta(minutes=1)  # now due
    assert _stopper(ctx, fake, minutes=5).tick() is None and fake.status == "RUNNING"
    assert _stopper(ctx, fake, minutes=0).tick() is None  # 0 = never stop it
    fake.started_at = (NOW - timedelta(minutes=2)).isoformat()
    with ctx.db.transaction() as s:
        s.get(Job, "job_later").status = "CANCELLED"
    assert _stopper(ctx, fake, minutes=5).tick() is None  # just started: give it time
    stopped = FakeRunPod("EXITED")
    assert _stopper(ctx, stopped, minutes=5).tick() is None and "POST" not in str(stopped.calls)


def test_the_idle_check_runs_at_most_every_few_minutes(ctx):
    fake = FakeRunPod("RUNNING")
    t = [0.0]
    stopper = IdleStopper(fake.client, 30, ctx.db.session, check_every_s=120,
                          now=lambda: NOW, clock=lambda: t[0])
    stopper.tick()
    stopper.tick()
    assert fake.calls.count("GET /v1/pods/pod123") == 1
    t[0] = 121
    stopper.tick()
    assert fake.calls.count("GET /v1/pods/pod123") == 2
