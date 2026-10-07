"""YouTube Data API v3 calls the studio makes, over plain ``httpx``.

Only what publishing needs: who am I (``channels.list mine``), resumable ``videos.insert``,
``thumbnails.set``. Quota costs follow the API's documented table so the ledger stays honest:
videos.insert 1600 units, thumbnails.set 50, channels.list 1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from rokkur_studio.youtube.oauth import OAuthClient, Token, TokenStore, refresh

log = logging.getLogger(__name__)

API = "https://www.googleapis.com/youtube/v3"
UPLOAD = "https://www.googleapis.com/upload/youtube/v3"
CHUNK = 8 * 1024 * 1024  # bytes per resumable PUT; a multiple of 256 KiB as Google requires
QUOTA = {"channels.list": 1, "videos.insert": 1600, "thumbnails.set": 50}


class YouTubeError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None,
                 reason: str | None = None) -> None:
        super().__init__(message)
        self.status, self.reason = status, reason

    def to_dict(self) -> dict[str, Any]:
        return {"message": str(self), "status": self.status, "reason": self.reason}


def _error(r: httpx.Response, what: str) -> YouTubeError:
    reason = None
    try:
        err = r.json().get("error", {})
        reason = (err.get("errors") or [{}])[0].get("reason") or err.get("status")
        message = err.get("message") or r.text[:300]
    except ValueError:
        message = r.text[:300]
    return YouTubeError(f"{what} failed ({r.status_code}): {message}", status=r.status_code,
                        reason=reason)


class YouTubeClient:
    def __init__(self, client: OAuthClient, store: TokenStore, *,
                 http: httpx.Client | None = None) -> None:
        self.client, self.store = client, store
        self.http = http or httpx.Client(timeout=httpx.Timeout(60, read=600))
        self._token: Token | None = None
        self.quota_used = 0

    def _auth(self) -> dict[str, str]:
        if self._token is None:
            self._token = self.store.load()
        if self._token.expired():
            self._token = refresh(self.http, self.client, self._token)
            self.store.save(self._token)
        return {"Authorization": f"Bearer {self._token.access_token}"}

    def _spend(self, method: str) -> None:
        self.quota_used += QUOTA[method]

    def my_channel(self) -> dict[str, Any]:
        """``{"id", "title", "custom_url"}`` for the signed-in account's channel."""
        r = self.http.get(f"{API}/channels", params={"part": "snippet", "mine": "true"},
                          headers=self._auth())
        self._spend("channels.list")
        if r.status_code != 200:
            raise _error(r, "channels.list")
        items = r.json().get("items") or []
        if not items:
            raise YouTubeError("this Google account has no YouTube channel")
        snippet = items[0].get("snippet", {})
        return {"id": items[0]["id"], "title": snippet.get("title", ""),
                "custom_url": snippet.get("customUrl")}

    def upload_video(self, path: Path, body: dict[str, Any], *,
                     content_type: str = "video/mp4") -> dict[str, Any]:
        """Resumable ``videos.insert``: one session request, then the file in chunks."""
        size = path.stat().st_size
        r = self.http.post(f"{UPLOAD}/videos", params={"uploadType": "resumable",
                                                        "part": "snippet,status"},
                           headers={**self._auth(), "X-Upload-Content-Type": content_type,
                                    "X-Upload-Content-Length": str(size)}, json=body)
        self._spend("videos.insert")
        if r.status_code != 200 or "Location" not in r.headers:
            raise _error(r, "videos.insert (start)")
        session_url = r.headers["Location"]
        sent = 0
        with open(path, "rb") as fh:
            while sent < size:
                chunk = fh.read(CHUNK)
                end = sent + len(chunk) - 1
                r = self.http.put(session_url, content=chunk, headers={
                    **self._auth(), "Content-Type": content_type,
                    "Content-Range": f"bytes {sent}-{end}/{size}"})
                if r.status_code == 308:  # resume incomplete: Google tells us where it is
                    rng = r.headers.get("Range")
                    sent = int(rng.split("-")[1]) + 1 if rng else end + 1
                    fh.seek(sent)
                    continue
                if r.status_code in (200, 201):
                    data = r.json()
                    if "id" not in data:
                        raise YouTubeError("upload finished but no video id was returned")
                    return data
                raise _error(r, "videos.insert (upload)")
        # Every byte is in but the final answer never came (a 308 after the last chunk): ask
        # the session for its state, which is how Google says to resume.
        r = self.http.put(session_url, headers={**self._auth(),
                                                "Content-Range": f"bytes */{size}"})
        if r.status_code in (200, 201) and "id" in r.json():
            return r.json()
        raise _error(r, "videos.insert (finish)")

    def set_thumbnail(self, video_id: str, path: Path) -> None:
        r = self.http.post(f"{UPLOAD}/thumbnails/set", params={"videoId": video_id,
                                                                "uploadType": "media"},
                           headers={**self._auth(), "Content-Type": "image/jpeg"},
                           content=path.read_bytes())
        self._spend("thumbnails.set")
        if r.status_code != 200:
            raise _error(r, "thumbnails.set")

    def close(self) -> None:
        self.http.close()


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"
