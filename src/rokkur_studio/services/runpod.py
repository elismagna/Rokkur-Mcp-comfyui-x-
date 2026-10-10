"""Turn the RunPod cloud GPU on and off (docs/cloud.md, "Start the cloud GPU with the studio").

Talks to RunPod's REST API (https://docs.runpod.io/api-reference): ``GET /pods/{id}``,
``POST /pods/{id}/start`` and ``POST /pods/{id}/stop``, with the API key as a bearer token.
The launcher (``scripts/launch.ps1``) starts the pod through ``rokkur-studio cloud-pod``, then
starts ComfyUI on it and opens the SSH tunnel from Windows. The dashboard shows whether the pod
is on and can stop it, and the worker stops it after ``cloud.auto_stop_idle_minutes`` without
cloud work, because a running pod bills by the hour whether it renders or not.
"""

from __future__ import annotations

import logging
import posixpath
import shlex
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from rokkur_studio.config import CloudSection, project_target
from rokkur_studio.db.models import Job, Project

log = logging.getLogger(__name__)

API_URL = "https://rest.runpod.io/v1"


class RunPodError(Exception):
    """RunPod refused or could not be reached; the message is safe to show (no API key)."""


@dataclass(frozen=True)
class PodState:
    status: str                    # RUNNING, EXITED, TERMINATED (RunPod's desiredStatus)
    ip: str = ""
    ssh_port: int | None = None    # the public port mapped to the pod's port 22
    price_per_hour: float | None = None
    name: str = ""
    gpu: str = ""
    started_at: datetime | None = None

    @property
    def running(self) -> bool:
        return self.status == "RUNNING"

    @property
    def reachable(self) -> bool:
        """Running with its public SSH address known (empty while the pod is still starting)."""
        return self.running and bool(self.ip) and bool(self.ssh_port)

    def label(self) -> str:
        what = " · ".join(p for p in (self.gpu or self.name, self.price_label()) if p)
        word = {"RUNNING": "on", "EXITED": "stopped", "TERMINATED": "deleted"}.get(
            self.status, self.status.lower())
        return f"Cloud GPU {word}" + (f" · {what}" if what else "")

    def price_label(self) -> str:
        return f"${self.price_per_hour:g}/h" if self.price_per_hour else ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "running": self.running, "ip": self.ip,
                "ssh_port": self.ssh_port, "price_per_hour": self.price_per_hour,
                "name": self.name, "gpu": self.gpu, "label": self.label(),
                "started_at": self.started_at.isoformat() if self.started_at else None}


def _gpu_name(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("displayName") or raw.get("id") or raw.get("name") or "")
    return str(raw or "")


def _price(raw: dict[str, Any]) -> float | None:
    for key in ("adjustedCostPerHr", "costPerHr"):  # costPerHr is a string in RunPod's example
        try:
            value = float(raw.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _started(raw: Any) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def parse_pod(raw: dict[str, Any]) -> PodState:
    ports = raw.get("portMappings") or {}
    ssh = ports.get("22") if isinstance(ports, dict) else None
    return PodState(status=str(raw.get("desiredStatus") or "UNKNOWN").upper(),
                    ip=str(raw.get("publicIp") or ""),
                    ssh_port=int(ssh) if ssh else None, price_per_hour=_price(raw),
                    name=str(raw.get("name") or ""), gpu=_gpu_name(raw.get("gpu")),
                    started_at=_started(raw["lastStartedAt"]) if raw.get("lastStartedAt") else None)


class RunPodClient:
    def __init__(self, api_key: str, pod_id: str, *, base_url: str = API_URL,
                 transport: httpx.BaseTransport | None = None, timeout_s: float = 20) -> None:
        self.pod_id = pod_id
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport,
                                  headers={"Authorization": f"Bearer {api_key}"})

    def close(self) -> None:
        self._http.close()

    def _call(self, method: str, path: str) -> httpx.Response:
        try:
            r = self._http.request(method, path)
        except httpx.HTTPError as exc:
            raise RunPodError(f"RunPod could not be reached: {exc.__class__.__name__}") from exc
        if r.status_code == 401:
            raise RunPodError("RunPod refused the API key (STUDIO_CLOUD__RUNPOD_API_KEY in .env).")
        if r.status_code in (400, 404) and method == "GET":
            raise RunPodError(f"RunPod has no pod '{self.pod_id}' (STUDIO_CLOUD__RUNPOD_POD_ID).")
        if not r.is_success:
            detail = r.text.strip()[:300] or r.reason_phrase
            raise RunPodError(f"RunPod answered {r.status_code}: {detail}")
        return r

    def state(self) -> PodState:
        return parse_pod(self._call("GET", f"/pods/{self.pod_id}").json())

    def start(self) -> None:
        self._call("POST", f"/pods/{self.pod_id}/start")

    def stop(self) -> None:
        self._call("POST", f"/pods/{self.pod_id}/stop")

    def wait_until_reachable(self, *, timeout_s: float = 600, poll_s: float = 5,
                             sleep: Callable[[float], None] | None = None,
                             clock: Callable[[], float] | None = None) -> PodState:
        """Poll until the pod runs and has a public SSH port (RunPod fills it in after boot)."""
        sleep, clock = sleep or time.sleep, clock or time.monotonic
        deadline = clock() + timeout_s
        while True:
            state = self.state()
            if state.reachable:
                return state
            if state.status == "TERMINATED":
                raise RunPodError("The pod was deleted on RunPod; create a new one (docs/cloud.md).")
            if clock() >= deadline:
                raise RunPodError(f"The pod did not come up within {timeout_s / 60:g} minutes "
                                  f"(RunPod status {state.status}).")
            sleep(poll_s)


def make_client(cloud: CloudSection,
                transport: httpx.BaseTransport | None = None) -> RunPodClient | None:
    if not cloud.pod_control:
        return None
    return RunPodClient(cloud.runpod_api_key.get_secret_value().strip(),
                        cloud.runpod_pod_id.strip(), transport=transport)


def connect_info(cloud: CloudSection, state: PodState) -> dict[str, Any]:
    """What the launcher needs to start ComfyUI on the pod and open the tunnel."""
    port = cloud.remote_comfy_port
    folder = cloud.remote_comfy_dir.rstrip("/") or "/workspace/ComfyUI"
    logfile = posixpath.join(posixpath.dirname(folder) or "/", "comfyui.log")
    # No double quotes: Windows PowerShell 5.1 does not pass them through to ssh.exe intact.
    remote_start = (
        f"curl -fs -o /dev/null http://127.0.0.1:{port}/system_stats || "
        f"(cd {shlex.quote(folder)} && setsid nohup python main.py --listen 127.0.0.1 "
        f"--port {port} > {shlex.quote(logfile)} 2>&1 < /dev/null &)")
    return {**state.to_dict(), "ssh_user": cloud.ssh_user, "ssh_key": cloud.ssh_key,
            "local_port": cloud.local_port, "remote_port": port, "remote_start": remote_start,
            "ask_on_launch": cloud.ask_on_launch,
            "auto_stop_idle_minutes": cloud.auto_stop_idle_minutes}


# -- the dashboard's view of the pod -----------------------------------------------------------
class PodStatusCache:
    """The pod's last known state for the dashboard rail, refreshed in the background.

    Every page polls ``/ui/status``; asking RunPod on each poll would slow every page down, so
    this returns the last answer at once and refreshes it at most every ``max_age_s``.
    """

    def __init__(self, fetch: Callable[[], PodState], *, max_age_s: float = 60,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._fetch, self.max_age_s, self._clock = fetch, max_age_s, clock
        self._lock = threading.Lock()
        self._state: PodState | None = None
        self._error = ""
        self._at = float("-inf")
        self._refreshing = False

    def get(self, *, wait: bool = False) -> tuple[PodState | None, str]:
        with self._lock:
            stale = self._clock() - self._at >= self.max_age_s
            start = stale and not self._refreshing
            if start:
                self._refreshing = True
        if start:
            if wait:
                self._refresh()
            else:
                threading.Thread(target=self._refresh, daemon=True, name="runpod-status").start()
        with self._lock:
            return self._state, self._error

    def invalidate(self) -> None:
        """Ask RunPod again on the next look (after the studio started or stopped the pod)."""
        with self._lock:
            self._at = float("-inf")

    def _refresh(self) -> None:
        try:
            state, error = self._fetch(), ""
        except RunPodError as exc:
            state, error = None, str(exc)
        except Exception as exc:  # never break the dashboard over the pod
            state, error = None, f"RunPod status failed: {exc.__class__.__name__}"
        with self._lock:
            self._state, self._error, self._at = state, error, self._clock()
            self._refreshing = False


# -- stop an idle pod --------------------------------------------------------------------------
def _uses_cloud(job: Job, creative: dict[str, Any] | None) -> bool:
    return (job.payload or {}).get("render_on") == "cloud" or project_target(creative) == "cloud"


def last_cloud_use(session: Session, now: datetime, window: timedelta) -> datetime | None:
    """When the studio last used the cloud GPU: ``now`` while a cloud job is due or running.

    Cloud jobs are image and sound jobs with ``render_on: cloud`` and every job of a video set to
    render on the cloud. A job scheduled for later (a retry wait, a timed release) does not count
    until it is due, so a waiting job cannot keep the pod billing.
    """
    rows = session.execute(
        select(Job, Project.creative_input)
        .outerjoin(Project, Project.id == Job.project_id)
        .where(or_(Job.status.in_(("QUEUED", "RUNNING", "RETRY_WAIT")),
                   Job.finished_at >= now - window))).all()
    last: datetime | None = None
    for job, creative in rows:
        if not _uses_cloud(job, creative):
            continue
        if job.status == "RUNNING" or (job.status in ("QUEUED", "RETRY_WAIT")
                                       and job.run_after <= now):
            return now
        if job.finished_at and (last is None or job.finished_at > last):
            last = job.finished_at
    return last


class IdleStopper:
    """Stops the pod after ``cloud.auto_stop_idle_minutes`` with no cloud work (worker loop)."""

    def __init__(self, client_factory: Callable[[], RunPodClient | None], idle_minutes: float,
                 db_session: Callable[[], Any], *, busy_probe: Callable[[], bool] | None = None,
                 check_every_s: float = 120,
                 now: Callable[[], datetime] = lambda: datetime.now(UTC),
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.client_factory, self.idle_minutes = client_factory, idle_minutes
        self.db_session, self.busy_probe = db_session, busy_probe
        self.check_every_s, self._now, self._clock = check_every_s, now, clock
        self._next = float("-inf")
        self._busy_seen: datetime | None = None

    def tick(self) -> str | None:
        """Check at most every ``check_every_s``; returns why it stopped the pod, if it did."""
        if self.idle_minutes <= 0 or self._clock() < self._next:
            return None
        self._next = self._clock() + self.check_every_s
        try:
            return self._check()
        except RunPodError as exc:
            log.warning("cloud GPU idle check failed: %s", exc)
        except Exception:
            log.warning("cloud GPU idle check failed", exc_info=True)
        return None

    def _check(self) -> str | None:
        client = self.client_factory()
        if client is None:
            return None
        try:
            state = client.state()
            if not state.running:
                return None
            now, window = self._now(), timedelta(minutes=self.idle_minutes)
            if self.busy_probe is not None and self.busy_probe():
                self._busy_seen = now  # someone is using the pod's ComfyUI directly
            with self.db_session() as s:
                used = last_cloud_use(s, now, window)
            latest = max((t for t in (used, self._busy_seen, state.started_at) if t),
                         default=None)
            if latest is not None and now - latest < window:
                return None
            client.stop()
        finally:
            client.close()
        reason = (f"Stopped the cloud GPU after {self.idle_minutes:g} minutes without cloud work "
                  f"(cloud.auto_stop_idle_minutes).")
        log.info(reason)
        return reason


def cloud_queue_busy(make_comfy: Callable[[], Any] | None) -> Callable[[], bool] | None:
    """A probe that is true while the cloud ComfyUI has prompts running or waiting."""
    if make_comfy is None:
        return None

    def probe() -> bool:
        try:
            client = make_comfy()
        except Exception:
            return False
        try:
            q = client.queue()
            return bool(q.get("queue_running") or q.get("queue_pending"))
        except Exception:
            return False  # tunnel closed or ComfyUI not started: nobody is using it
        finally:
            client.close()
    return probe
