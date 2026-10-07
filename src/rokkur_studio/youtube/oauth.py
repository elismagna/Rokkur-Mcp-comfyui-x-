"""Google OAuth for an installed app, with only ``httpx``: no Google SDK, no hidden state.

* The OAuth client (``client_id`` / ``client_secret``) is the JSON Google Cloud hands out for a
  "Desktop app" client. It lives in ``secrets/`` (git-ignored) and is read, never copied.
* The refresh token is written to ``secrets/youtube_token.json`` with owner-only permissions.
  It never enters the database, the logs or an event.
* Consent uses the loopback redirect (``http://127.0.0.1:<port>/``) plus PKCE, as Google
  recommends for native apps; the out-of-band copy/paste flow is gone.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = ("https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube.readonly")
# Refresh this long before the access token actually expires.
REFRESH_MARGIN_S = 120


class OAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class OAuthClient:
    client_id: str
    client_secret: str
    token_uri: str = TOKEN_URL

    @classmethod
    def load(cls, path: Path) -> OAuthClient:
        if not path.is_file():
            raise OAuthError(f"OAuth client file not found: {path} (download the Desktop-app "
                             "client JSON from Google Cloud console; see docs/youtube.md)")
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get("installed") or data.get("web")
        if not block or "client_id" not in block or "client_secret" not in block:
            raise OAuthError(f"{path} is not a Google OAuth client file (no 'installed' block)")
        return cls(block["client_id"], block["client_secret"], block.get("token_uri", TOKEN_URL))


@dataclass
class Token:
    refresh_token: str
    access_token: str = ""
    expires_at: float = 0.0
    scope: str = ""

    def expired(self) -> bool:
        return not self.access_token or time.time() >= self.expires_at - REFRESH_MARGIN_S


class TokenStore:
    """A JSON file holding one refresh token, owner-readable only."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> Token:
        if not self.exists():
            raise OAuthError(f"not signed in to YouTube ({self.path} missing); "
                             "run: rokkur-studio youtube-auth")
        return Token(**json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, token: Token) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8", opener=lambda p, f: os.open(p, f, 0o600)) as fh:
            json.dump(token.__dict__, fh)
        os.replace(tmp, self.path)
        with contextlib.suppress(OSError):  # Windows/Docker bind mounts may refuse
            os.chmod(self.path, 0o600)

    def delete(self) -> None:
        if self.exists():
            self.path.unlink()


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_auth_url(client: OAuthClient, redirect_uri: str, *, state: str,
                   code_challenge: str) -> str:
    return AUTH_URL + "?" + urlencode({
        "client_id": client.client_id, "redirect_uri": redirect_uri, "response_type": "code",
        "scope": " ".join(SCOPES), "access_type": "offline", "prompt": "consent",
        "include_granted_scopes": "true", "state": state,
        "code_challenge": code_challenge, "code_challenge_method": "S256",
    })


def _token_request(http: httpx.Client, client: OAuthClient, form: dict[str, str]) -> dict[str, Any]:
    try:
        r = http.post(client.token_uri, data={**form, "client_id": client.client_id,
                                              "client_secret": client.client_secret})
    except httpx.HTTPError as exc:
        raise OAuthError(f"token request failed: {exc}") from exc
    if r.status_code != 200:
        detail = r.json() if r.headers.get("content-type", "").startswith("application/json") \
            else r.text[:300]
        raise OAuthError(f"token request rejected ({r.status_code}): {detail}")
    return r.json()


def exchange_code(http: httpx.Client, client: OAuthClient, *, code: str, redirect_uri: str,
                  code_verifier: str) -> Token:
    data = _token_request(http, client, {"grant_type": "authorization_code", "code": code,
                                         "redirect_uri": redirect_uri,
                                         "code_verifier": code_verifier})
    if "refresh_token" not in data:
        raise OAuthError("Google returned no refresh token; revoke the app's access at "
                         "myaccount.google.com/permissions and sign in again")
    return Token(refresh_token=data["refresh_token"], access_token=data["access_token"],
                 expires_at=time.time() + float(data.get("expires_in", 3600)),
                 scope=data.get("scope", ""))


def refresh(http: httpx.Client, client: OAuthClient, token: Token) -> Token:
    data = _token_request(http, client, {"grant_type": "refresh_token",
                                         "refresh_token": token.refresh_token})
    token.access_token = data["access_token"]
    token.expires_at = time.time() + float(data.get("expires_in", 3600))
    token.scope = data.get("scope", token.scope)
    return token


def parse_redirect(url_or_query: str, *, expected_state: str) -> str:
    """Pull the authorization code out of the URL the browser was sent back to."""
    parsed = urlparse(url_or_query if "?" in url_or_query else "?" + url_or_query)
    q = parse_qs(parsed.query)
    if "error" in q:
        raise OAuthError(f"consent denied: {q['error'][0]}")
    if q.get("state", [None])[0] != expected_state:
        raise OAuthError("state mismatch: the redirect did not belong to this sign-in attempt")
    if not q.get("code"):
        raise OAuthError("no authorization code in the redirect URL")
    return q["code"][0]


class _Catcher(BaseHTTPRequestHandler):
    query: str | None = None

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        type(self).query = urlparse(self.path).query
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write("<h2>Rökkur Studio: signed in. You can close this tab.</h2>"
                         .encode())

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return  # keep the auth code out of the console


def wait_for_redirect(port: int, *, timeout_s: float, bind: str = "0.0.0.0") -> str:
    """Serve one request on the loopback port and return its query string."""
    _Catcher.query = None
    server = HTTPServer((bind, port), _Catcher)
    server.timeout = timeout_s
    try:
        server.handle_request()
    finally:
        server.server_close()
    if _Catcher.query is None:
        raise OAuthError(f"no redirect received on port {port} within {timeout_s:.0f}s")
    return _Catcher.query


def installed_flow(client: OAuthClient, store: TokenStore, *, port: int, http: httpx.Client,
                   open_url: Any, timeout_s: float = 300, paste: bool = False,
                   read_line: Any = input) -> Token:
    """Run consent end to end. ``open_url(url)`` shows the user the link; ``paste`` instead of
    listening reads the redirected URL from ``read_line()`` (for hosts where the port cannot
    be reached, e.g. Docker without the port published)."""
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    redirect_uri = f"http://127.0.0.1:{port}/"
    open_url(build_auth_url(client, redirect_uri, state=state, code_challenge=challenge))
    raw = read_line() if paste else wait_for_redirect(port, timeout_s=timeout_s)
    code = parse_redirect(raw.strip(), expected_state=state)
    token = exchange_code(http, client, code=code, redirect_uri=redirect_uri,
                          code_verifier=verifier)
    store.save(token)
    return token
