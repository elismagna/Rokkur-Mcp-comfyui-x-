"""An in-memory Google: token endpoint, channels.list, resumable videos.insert, thumbnails.set,
playlists.list (paged) and playlistItems.insert."""

from __future__ import annotations

import json
from typing import Any

import httpx


class FakeGoogle:
    PLAYLISTS = [{"id": "PLshorts0001", "title": "Shorts", "count": 12, "privacy": "public"},
                 {"id": "PLclaymation1", "title": "claymation", "count": 3, "privacy": "public"},
                 {"id": "PLdrafts00001", "title": "Drafts", "count": 0, "privacy": "private"}]

    def __init__(self, *, chunk_308: bool = False, fail_upload: int | None = None,
                 fail_thumbnail: bool = False, fail_playlist_item: bool = False,
                 playlist_page_size: int = 50) -> None:
        self.tokens: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.received = bytearray()
        self.thumbnails: list[str] = []
        self.playlist_items: list[dict[str, Any]] = []
        self.playlist_pages = 0
        self.chunk_308, self.fail_upload, self.fail_thumbnail = chunk_308, fail_upload, fail_thumbnail
        self.fail_playlist_item, self.playlist_page_size = fail_playlist_item, playlist_page_size
        self.refreshes = 0
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        url, path = request.url, request.url.path
        if url.host == "oauth2.googleapis.com" and path == "/token":
            form = dict(p.split("=", 1) for p in request.content.decode().split("&"))
            self.tokens.append(form)
            if form["grant_type"] == "refresh_token":
                self.refreshes += 1
                return httpx.Response(200, json={"access_token": "at-refreshed",
                                                 "expires_in": 3599})
            return httpx.Response(200, json={"access_token": "at-1", "refresh_token": "rt-1",
                                             "expires_in": 3599, "scope": "upload"})
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer at-"):
            return httpx.Response(401, json={"error": {"message": "no token",
                                                       "errors": [{"reason": "authError"}]}})
        if path == "/youtube/v3/channels":
            return httpx.Response(200, json={"items": [{"id": "UC123", "snippet": {
                "title": "Rökkur", "customUrl": "@rokkur"}}]})
        if path == "/upload/youtube/v3/videos" and request.method == "POST":
            if self.fail_upload == 0:
                return httpx.Response(403, json={"error": {"message": "quota",
                                                           "errors": [{"reason": "quotaExceeded"}]}})
            self.uploads.append({"body": json.loads(request.content),
                                 "length": int(request.headers["X-Upload-Content-Length"])})
            self.received = bytearray()
            return httpx.Response(200, headers={"Location": "https://www.googleapis.com/up/1"})
        if path == "/up/1" and request.method == "PUT":
            if request.headers["Content-Range"].startswith("bytes */"):  # status query
                return httpx.Response(200, json={"id": "vid123"})
            start, end, total = _range(request.headers["Content-Range"])
            if self.fail_upload == 1:
                return httpx.Response(500, json={"error": {"message": "backend"}})
            self.received[start:end + 1] = request.content
            if end + 1 < total:
                return httpx.Response(308, headers={"Range": f"bytes=0-{end}"})
            if self.chunk_308:  # Google may ask for one more round trip after the last chunk
                self.chunk_308 = False
                return httpx.Response(308, headers={"Range": f"bytes=0-{end}"})
            return httpx.Response(200, json={"id": "vid123", "status": {
                "privacyStatus": self.uploads[-1]["body"]["status"]["privacyStatus"]}})
        if path == "/youtube/v3/playlists" and request.method == "GET":
            assert url.params["mine"] == "true"
            self.playlist_pages += 1
            start = int(url.params.get("pageToken") or 0)
            end = start + self.playlist_page_size
            page = {"items": [{"id": p["id"], "snippet": {"title": p["title"]},
                               "contentDetails": {"itemCount": p["count"]},
                               "status": {"privacyStatus": p["privacy"]}}
                              for p in self.PLAYLISTS[start:end]]}
            if end < len(self.PLAYLISTS):
                page["nextPageToken"] = str(end)
            return httpx.Response(200, json=page)
        if path == "/youtube/v3/playlistItems" and request.method == "POST":
            if self.fail_playlist_item:
                return httpx.Response(404, json={"error": {
                    "message": "playlist not found", "errors": [{"reason": "playlistNotFound"}]}})
            self.playlist_items.append(json.loads(request.content)["snippet"])
            return httpx.Response(200, json={"id": f"item{len(self.playlist_items)}"})
        if path == "/upload/youtube/v3/thumbnails/set":
            if self.fail_thumbnail:
                return httpx.Response(400, json={"error": {"message": "bad image"}})
            self.thumbnails.append(url.params["videoId"])
            return httpx.Response(200, json={"items": []})
        return httpx.Response(404, json={"error": f"unhandled {request.method} {url}"})


def _range(header: str) -> tuple[int, int, int]:
    span, total = header.removeprefix("bytes ").split("/")
    start, end = span.split("-")
    return int(start), int(end), int(total)
