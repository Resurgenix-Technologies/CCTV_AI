"""Dependency-free, person-scoped demo portal for observed activity.

The server intentionally has a narrow scope.  An opaque QR credential is used
once to create a short-lived in-memory session; subsequent requests carry only
a new random session cookie.  Access logging is disabled because the QR route
contains a bearer credential and request headers contain the session secret.

This is an HTTP application server, not a TLS terminator.  Put it behind an
HTTPS reverse proxy and enable secure cookies outside local demonstrations.
Sessions are process-local and disappear on restart.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import secrets
import stat
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlsplit

from src.activity_repository import (
    ActivityRepository,
    ActivityRepositoryError,
    InvalidPortalCredentialError,
    PersonNotFoundError,
    PortalSummary,
    PortalTimelineEvent,
)


SESSION_COOKIE_NAME = "cctv_portal_session"
DEFAULT_SESSION_TTL_SECONDS = 15 * 60
DEFAULT_MAX_SESSIONS = 10_000

_OPAQUE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{20,200}\Z")
_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    "img-src 'self'; "
    "style-src 'unsafe-inline'; "
    "connect-src 'self'; "
    "form-action 'self'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)

_PAGE_STYLE = """
:root {
  color-scheme: light;
  --ink: #12221d;
  --muted: #62716b;
  --line: #dbe4df;
  --paper: #ffffff;
  --wash: #f2f7f4;
  --brand: #116149;
  --brand-dark: #0a4534;
  --accent: #d9f3e8;
  --warning: #8a4b08;
  --shadow: 0 12px 34px rgba(20, 54, 43, 0.09);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: linear-gradient(150deg, #e9f5ef 0, #f8faf9 42%, #eef3f0 100%);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  line-height: 1.5;
}
button, input { font: inherit; }
.shell { width: min(1120px, calc(100% - 28px)); margin: 0 auto; }
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 20px 0;
}
.brand { display: flex; align-items: center; gap: 10px; font-weight: 800; }
.brand-mark {
  width: 36px; height: 36px; border-radius: 11px; display: grid;
  place-items: center; color: white; background: var(--brand);
  box-shadow: var(--shadow);
}
.logout {
  border: 1px solid #b9c9c1; border-radius: 999px; background: rgba(255,255,255,.8);
  color: var(--brand-dark); padding: 8px 15px; cursor: pointer; font-weight: 700;
}
.hero {
  display: grid; grid-template-columns: auto 1fr; gap: 20px; align-items: center;
  padding: 28px; border-radius: 24px; background: var(--brand-dark); color: white;
  box-shadow: var(--shadow); overflow: hidden; position: relative;
}
.hero::after {
  content: ""; position: absolute; width: 220px; height: 220px; right: -80px;
  top: -110px; border-radius: 50%; background: rgba(80, 206, 157, .18);
}
.avatar {
  width: 92px; height: 92px; border-radius: 24px; object-fit: cover;
  border: 3px solid rgba(255,255,255,.75); background: #b9d5c9;
  position: relative; z-index: 1;
}
.avatar-fallback {
  display: grid; place-items: center; color: var(--brand-dark);
  font-size: 34px; font-weight: 900;
}
.eyebrow {
  margin: 0 0 4px; color: #bfead8; font-size: .78rem; font-weight: 800;
  letter-spacing: .12em; text-transform: uppercase;
}
h1 { margin: 0; font-size: clamp(1.7rem, 5vw, 2.65rem); line-height: 1.08; }
.hero-note { margin: 8px 0 0; color: #dcece5; }
.metrics {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px;
  margin: 18px 0;
}
.metric, .panel {
  background: rgba(255,255,255,.92); border: 1px solid rgba(201,216,208,.9);
  box-shadow: var(--shadow);
}
.metric { border-radius: 18px; padding: 18px; }
.metric-label { margin: 0; color: var(--muted); font-size: .84rem; font-weight: 700; }
.metric-value { margin: 5px 0 0; font-size: clamp(1.35rem, 4vw, 2rem); font-weight: 900; }
.layout {
  display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(280px, .85fr);
  gap: 18px; align-items: start; padding-bottom: 38px;
}
.stack { display: grid; gap: 18px; }
.panel { border-radius: 20px; padding: 20px; }
.panel-head {
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  margin-bottom: 13px;
}
h2 { margin: 0; font-size: 1.08rem; }
.subtle { color: var(--muted); font-size: .82rem; }
.list { display: grid; gap: 10px; }
.row {
  border: 1px solid var(--line); border-radius: 14px; padding: 13px 14px;
  background: var(--paper);
}
.row-top { display: flex; justify-content: space-between; gap: 12px; align-items: start; }
.row-title { margin: 0; font-weight: 800; }
.row-meta { margin: 4px 0 0; color: var(--muted); font-size: .86rem; }
.duration { white-space: nowrap; color: var(--brand); font-weight: 900; }
.pill {
  display: inline-flex; align-items: center; border-radius: 999px;
  padding: 3px 8px; background: var(--accent); color: var(--brand-dark);
  font-size: .72rem; font-weight: 800;
}
.pill.live { background: #fff1d6; color: var(--warning); }
.empty {
  border: 1px dashed #bdcbc4; border-radius: 14px; padding: 20px;
  color: var(--muted); text-align: center;
}
.notice {
  margin: 0 0 18px; border-left: 4px solid var(--brand); border-radius: 10px;
  padding: 11px 14px; background: #eef8f3; color: #36554a; font-size: .86rem;
}
.error-card {
  width: min(520px, calc(100% - 32px)); margin: 12vh auto; padding: 28px;
  border: 1px solid var(--line); border-radius: 20px; background: white;
  box-shadow: var(--shadow); text-align: center;
}
.error-card h1 { font-size: 1.6rem; }
.error-card p { color: var(--muted); }
@media (max-width: 760px) {
  .metrics { grid-template-columns: 1fr; }
  .layout { grid-template-columns: 1fr; }
  .hero { grid-template-columns: 68px 1fr; padding: 21px; }
  .avatar { width: 68px; height: 68px; border-radius: 18px; }
}
@media (max-width: 420px) {
  .shell { width: min(100% - 18px, 1120px); }
  .topbar { padding: 13px 0; }
  .brand span:last-child { display: none; }
  .hero { grid-template-columns: 1fr; }
  .row-top { display: grid; }
  .duration { justify-self: start; }
}
"""


@dataclass(frozen=True)
class _Session:
    """Server-side session state; the raw cookie is never retained."""

    person_id: str
    created_at: float
    expires_at: float


class SessionStore:
    """Thread-safe, bounded, expiring in-memory session storage."""

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], float] = time.time,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive.")
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive.")

        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._clock = clock
        self._token_factory = token_factory or (
            lambda: secrets.token_urlsafe(32)
        )
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _purge_expired(self, now: float) -> None:
        expired = [
            key
            for key, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for key in expired:
            self._sessions.pop(key, None)

    def create(self, person_id: str) -> str:
        """Create a fixed-lifetime session and return its cookie secret."""

        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            if len(self._sessions) >= self.max_sessions:
                oldest_key = min(
                    self._sessions,
                    key=lambda key: self._sessions[key].created_at,
                )
                self._sessions.pop(oldest_key, None)

            for _ in range(8):
                token = self._token_factory()
                if not token:
                    continue
                key = self._digest(token)
                if key not in self._sessions:
                    self._sessions[key] = _Session(
                        person_id=person_id,
                        created_at=now,
                        expires_at=now + self.ttl_seconds,
                    )
                    return token

        raise RuntimeError("Could not allocate a unique portal session.")

    def resolve(self, token: str) -> str | None:
        """Return the owner of an active session without extending it."""

        if not token:
            return None
        now = self._clock()
        key = self._digest(token)
        with self._lock:
            self._purge_expired(now)
            session = self._sessions.get(key)
            return session.person_id if session is not None else None

    def destroy(self, token: str | None) -> None:
        """Forget a session if a cookie was supplied."""

        if not token:
            return
        key = self._digest(token)
        with self._lock:
            self._sessions.pop(key, None)


class PortalHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying portal dependencies for request handlers."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        repository: ActivityRepository,
        sessions: SessionStore,
        *,
        secure_cookie: bool,
        photo_root: Path,
    ) -> None:
        self.repository = repository
        self.sessions = sessions
        self.secure_cookie = secure_cookie
        self.photo_root = photo_root.resolve()
        super().__init__(server_address, PortalRequestHandler)


def create_portal_server(
    server_address: tuple[str, int],
    repository: ActivityRepository,
    *,
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    secure_cookie: bool = False,
    photo_root: Path | None = None,
    clock: Callable[[], float] = time.time,
) -> PortalHTTPServer:
    """Build a configured portal server without starting its event loop."""

    sessions = SessionStore(
        session_ttl_seconds,
        max_sessions=max_sessions,
        clock=clock,
    )
    return PortalHTTPServer(
        server_address,
        repository,
        sessions,
        secure_cookie=secure_cookie,
        photo_root=photo_root or Path.cwd(),
    )


class PortalRequestHandler(BaseHTTPRequestHandler):
    """Serve the QR exchange, personal views, and liveness response."""

    protocol_version = "HTTP/1.1"
    server_version = "CCTVPortal"
    sys_version = ""
    server: PortalHTTPServer

    def version_string(self) -> str:
        return self.server_version

    def log_message(self, format: str, *args: object) -> None:
        """Disable access logs so bearer credentials can never enter them."""

        return

    def do_GET(self) -> None:
        try:
            self._dispatch_get()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            # Do not print a traceback containing request state or credentials.
            self._send_internal_error_safely()

    def do_POST(self) -> None:
        try:
            parsed = urlsplit(self.path)
            if parsed.path == "/logout" and not parsed.query:
                self._discard_small_request_body()
                self._logout()
                return
            self._method_not_allowed()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self._send_internal_error_safely()

    def _dispatch_get(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path

        if path.startswith("/q/"):
            if parsed.query:
                self._not_found()
            else:
                self._exchange_qr(path[len("/q/") :])
            return
        if parsed.query:
            self._not_found()
            return
        if path == "/portal":
            self._portal()
        elif path == "/api/v1/portal/me":
            self._portal_api()
        elif path == "/portal/photo":
            self._portal_photo()
        elif path == "/logout":
            self._method_not_allowed(allow="POST")
        elif path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._not_found()

    def _exchange_qr(self, raw_segment: str) -> None:
        try:
            token = unquote(raw_segment, errors="strict")
        except (UnicodeDecodeError, ValueError):
            self._not_found()
            return

        if not _OPAQUE_TOKEN_PATTERN.fullmatch(token):
            self._not_found()
            return

        try:
            person_id = self.server.repository.resolve_portal_credential(token)
        except InvalidPortalCredentialError:
            self._not_found()
            return
        except (ActivityRepositoryError, ValueError):
            self._service_unavailable()
            return

        old_session = self._session_cookie()
        self.server.sessions.destroy(old_session)
        session_token = self.server.sessions.create(person_id)
        self._redirect(
            "/portal",
            set_cookie=self._session_cookie_header(session_token),
        )

    def _portal(self) -> None:
        activity = self._authenticated_activity()
        if activity is None:
            return
        summary, timeline = activity
        body = _render_dashboard(summary, timeline).encode("utf-8")
        self._send_bytes(200, "text/html; charset=utf-8", body)

    def _portal_api(self) -> None:
        activity = self._authenticated_activity(api=True)
        if activity is None:
            return
        summary, timeline = activity
        self._send_json(200, _portal_payload(summary, timeline))

    def _portal_photo(self) -> None:
        identity = self._authenticated_person()
        if identity is None:
            return
        _, person_id = identity

        try:
            summary = self.server.repository.get_portal_summary(person_id)
        except PersonNotFoundError:
            self._invalidate_session(api=False)
            return
        except ActivityRepositoryError:
            self._service_unavailable()
            return

        photo = summary.profile_photo
        if photo is None:
            self._not_found()
            return

        configured = Path(photo.path)
        candidate = (
            configured
            if configured.is_absolute()
            else self.server.photo_root / configured
        )

        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.server.photo_root)
            stream = resolved.open("rb")
        except (OSError, RuntimeError, ValueError):
            self._not_found()
            return

        with stream:
            try:
                details = stream.seek(0) or stream.read(16)
                if isinstance(details, int):
                    details = stream.read(16)
                stream.seek(0)
                file_stat = stream.fileno()
                file_info = __import__("os").fstat(file_stat)
                if not stat.S_ISREG(file_info.st_mode):
                    self._not_found()
                    return
                content_type, suffix = _image_type(details)
                if content_type is None:
                    self._not_found()
                    return
                self._start_response(
                    200,
                    content_type,
                    file_info.st_size,
                    extra_headers={
                        "Content-Disposition": (
                            f'inline; filename="profile.{suffix}"'
                        ),
                    },
                )
                while True:
                    chunk = stream.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except (OSError, ValueError):
                return

    def _authenticated_activity(
        self,
        *,
        api: bool = False,
    ) -> tuple[PortalSummary, tuple[PortalTimelineEvent, ...]] | None:
        identity = self._authenticated_person(api=api)
        if identity is None:
            return None
        _, person_id = identity
        try:
            summary = self.server.repository.get_portal_summary(person_id)
            timeline = self.server.repository.get_portal_timeline(person_id)
            return summary, timeline
        except PersonNotFoundError:
            self._invalidate_session(api=api)
            return None
        except ActivityRepositoryError:
            self._service_unavailable(api=api)
            return None

    def _authenticated_person(
        self,
        *,
        api: bool = False,
    ) -> tuple[str, str] | None:
        session_token = self._session_cookie()
        if session_token is None:
            self._unauthorized(api=api)
            return None
        person_id = self.server.sessions.resolve(session_token)
        if person_id is None:
            self._unauthorized(api=api, clear_cookie=True)
            return None
        return session_token, person_id

    def _invalidate_session(self, *, api: bool) -> None:
        session_token = self._session_cookie()
        self.server.sessions.destroy(session_token)
        self._unauthorized(api=api, clear_cookie=True)

    def _logout(self) -> None:
        self.server.sessions.destroy(self._session_cookie())
        body = _simple_page(
            "Signed out",
            "Your local portal session has ended. You may close this page.",
        ).encode("utf-8")
        self._send_bytes(
            200,
            "text/html; charset=utf-8",
            body,
            extra_headers={"Set-Cookie": self._clear_cookie_header()},
        )

    def _session_cookie(self) -> str | None:
        values = self.headers.get_all("Cookie", failobj=[]) or []
        if not values:
            return None
        jar = SimpleCookie()
        try:
            for value in values:
                jar.load(value)
        except CookieError:
            return None
        morsel = jar.get(SESSION_COOKIE_NAME)
        return morsel.value if morsel is not None else None

    def _session_cookie_header(self, token: str) -> str:
        attributes = [
            f"{SESSION_COOKIE_NAME}={token}",
            "Path=/",
            f"Max-Age={self.server.sessions.ttl_seconds}",
            "HttpOnly",
            "SameSite=Strict",
        ]
        if self.server.secure_cookie:
            attributes.append("Secure")
        return "; ".join(attributes)

    def _clear_cookie_header(self) -> str:
        attributes = [
            f"{SESSION_COOKIE_NAME}=",
            "Path=/",
            "Max-Age=0",
            "Expires=Thu, 01 Jan 1970 00:00:00 GMT",
            "HttpOnly",
            "SameSite=Strict",
        ]
        if self.server.secure_cookie:
            attributes.append("Secure")
        return "; ".join(attributes)

    def _unauthorized(
        self,
        *,
        api: bool,
        clear_cookie: bool = False,
    ) -> None:
        headers = (
            {"Set-Cookie": self._clear_cookie_header()}
            if clear_cookie
            else None
        )
        if api:
            self._send_json(
                401,
                {"error": "authentication_required"},
                extra_headers=headers,
            )
            return
        body = _simple_page(
            "Authentication required",
            "Scan your issued QR credential to open your personal dashboard.",
        ).encode("utf-8")
        self._send_bytes(
            401,
            "text/html; charset=utf-8",
            body,
            extra_headers=headers,
        )

    def _not_found(self) -> None:
        body = _simple_page(
            "Not found",
            "The requested page is unavailable.",
        ).encode("utf-8")
        self._send_bytes(404, "text/html; charset=utf-8", body)

    def _service_unavailable(self, *, api: bool = False) -> None:
        if api:
            self._send_json(503, {"error": "service_unavailable"})
            return
        body = _simple_page(
            "Temporarily unavailable",
            "The activity service could not complete this request.",
        ).encode("utf-8")
        self._send_bytes(503, "text/html; charset=utf-8", body)

    def _method_not_allowed(self, *, allow: str = "GET") -> None:
        body = _simple_page(
            "Method not allowed",
            "This endpoint does not support that request method.",
        ).encode("utf-8")
        self._send_bytes(
            405,
            "text/html; charset=utf-8",
            body,
            extra_headers={"Allow": allow},
        )

    def _redirect(self, location: str, *, set_cookie: str) -> None:
        self._start_response(
            303,
            "text/plain; charset=utf-8",
            0,
            extra_headers={
                "Location": location,
                "Set-Cookie": set_cookie,
            },
        )

    def _send_json(
        self,
        status_code: int,
        payload: object,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(
            status_code,
            "application/json; charset=utf-8",
            body,
            extra_headers=extra_headers,
        )

    def _send_bytes(
        self,
        status_code: int,
        content_type: str,
        body: bytes,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._start_response(
            status_code,
            content_type,
            len(body),
            extra_headers=extra_headers,
        )
        self.wfile.write(body)

    def _start_response(
        self,
        status_code: int,
        content_type: str,
        content_length: int,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Security-Policy", _CONTENT_SECURITY_POLICY)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        if self.server.secure_cookie:
            self.send_header(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()

    def _discard_small_request_body(self) -> None:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            length = 0
        if 0 < length <= 4096:
            self.rfile.read(length)
        elif length > 4096:
            self.close_connection = True

    def _send_internal_error_safely(self) -> None:
        try:
            body = _simple_page(
                "Request failed",
                "The server could not complete this request.",
            ).encode("utf-8")
            self._send_bytes(500, "text/html; charset=utf-8", body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def _image_type(header: bytes) -> tuple[str | None, str | None]:
    """Recognize a small allowlist of non-SVG image formats by signature."""

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", "gif"
    if (
        len(header) >= 12
        and header.startswith(b"RIFF")
        and header[8:12] == b"WEBP"
    ):
        return "image/webp", "webp"
    return None, None


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def _format_timestamp(value: str | None) -> str:
    if value is None:
        return "In progress"
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        utc_value = parsed.astimezone(timezone.utc)
        return utc_value.strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return value


def _escaped(value: object | None) -> str:
    if value is None:
        return "—"
    return html.escape(str(value), quote=True)


def _render_dashboard(
    summary: PortalSummary,
    timeline: tuple[PortalTimelineEvent, ...],
) -> str:
    name = _escaped(summary.full_name)
    first_character = _escaped(summary.full_name[:1].upper() or "?")
    if summary.profile_photo is not None:
        avatar = (
            '<img class="avatar" src="/portal/photo" alt="Profile photo of '
            + name
            + '">'
        )
        photo_note = (
            "Photo captured "
            + _escaped(_format_timestamp(summary.profile_photo.captured_at))
            if summary.profile_photo.captured_at
            else "Profile photo available"
        )
    else:
        avatar = (
            '<div class="avatar avatar-fallback" aria-label="No profile photo">'
            + first_character
            + "</div>"
        )
        photo_note = "No profile photo is configured"

    visits = "".join(_visit_html(visit) for visit in summary.visits)
    if not visits:
        visits = '<div class="empty">No observed visits are available.</div>'

    interactions = "".join(
        (
            '<article class="row"><div class="row-top"><div>'
            '<p class="row-title">'
            + _escaped(item.counterpart_full_name)
            + '</p><p class="row-meta">'
            + f"{item.interaction_count} estimated episode"
            + ("s" if item.interaction_count != 1 else "")
            + " · Last observed "
            + _escaped(_format_timestamp(item.last_interaction_at))
            + '</p></div><span class="duration">'
            + _escaped(_format_duration(item.total_duration_seconds))
            + "</span></div></article>"
        )
        for item in summary.interactions
    )
    if not interactions:
        interactions = (
            '<div class="empty">No estimated proximity interactions are '
            "available.</div>"
        )

    timeline_rows = "".join(_timeline_html(event) for event in timeline)
    if not timeline_rows:
        timeline_rows = '<div class="empty">No timeline events are available.</div>'

    return (
        "<!doctype html><html lang=\"en\"><head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="referrer" content="no-referrer">'
        "<title>My activity · CCTV Portal</title>"
        "<style>"
        + _PAGE_STYLE
        + "</style></head><body>"
        '<div class="shell"><header class="topbar">'
        '<div class="brand"><span class="brand-mark">C</span>'
        "<span>Personal activity portal</span></div>"
        '<form action="/logout" method="post"><button class="logout" '
        'type="submit">Sign out</button></form></header>'
        '<main><section class="hero">'
        + avatar
        + '<div><p class="eyebrow">Your observed activity</p><h1>'
        + name
        + '</h1><p class="hero-note">'
        + _escaped(photo_note)
        + "</p></div></section>"
        '<section class="metrics" aria-label="Activity totals">'
        + _metric_html(
            "Observed presence",
            _format_duration(summary.total_presence_seconds),
        )
        + _metric_html(
            "Estimated interaction time",
            _format_duration(summary.total_interaction_seconds),
        )
        + _metric_html("Visits", str(len(summary.visits)))
        + "</section>"
        '<p class="notice"><strong>How to read this:</strong> Presence is based '
        "on stored entry and exit observations. Interaction records estimate "
        "sustained visual proximity; they do not prove that a conversation took "
        "place.</p>"
        '<section class="layout"><div class="stack">'
        '<section class="panel"><div class="panel-head"><h2>Visit history</h2>'
        '<span class="subtle">Newest first</span></div><div class="list">'
        + visits
        + "</div></section>"
        '<section class="panel"><div class="panel-head"><h2>Activity timeline</h2>'
        '<span class="subtle">Observed events</span></div><div class="list">'
        + timeline_rows
        + "</div></section></div>"
        '<aside class="panel"><div class="panel-head"><h2>People nearby</h2>'
        '<span class="subtle">Estimated totals</span></div><div class="list">'
        + interactions
        + "</div></aside></section></main></div></body></html>"
    )


def _metric_html(label: str, value: str) -> str:
    return (
        '<article class="metric"><p class="metric-label">'
        + _escaped(label)
        + '</p><p class="metric-value">'
        + _escaped(value)
        + "</p></article>"
    )


def _visit_html(visit: object) -> str:
    entered_at = getattr(visit, "entered_at")
    exited_at = getattr(visit, "exited_at")
    duration_seconds = getattr(visit, "duration_seconds")
    entry_camera = getattr(visit, "entry_camera_source")
    exit_camera = getattr(visit, "exit_camera_source")
    camera_text = "Entry source: " + _escaped(entry_camera)
    if exit_camera:
        camera_text += " · Exit source: " + _escaped(exit_camera)
    status = (
        '<span class="pill live">In progress</span>'
        if exited_at is None
        else '<span class="pill">Completed</span>'
    )
    return (
        '<article class="row"><div class="row-top"><div><p class="row-title">'
        + _escaped(_format_timestamp(entered_at))
        + " → "
        + _escaped(_format_timestamp(exited_at))
        + '</p><p class="row-meta">'
        + camera_text
        + "</p></div><div>"
        + status
        + '<div class="duration">'
        + _escaped(_format_duration(duration_seconds))
        + "</div></div></div></article>"
    )


def _timeline_html(event: PortalTimelineEvent) -> str:
    if event.event_type == "PROXIMITY_INTERACTION":
        title = "Estimated proximity with " + _escaped(
            event.counterpart_full_name
        )
        context_parts = [
            _escaped(part)
            for part in (event.zone_label, event.camera_source)
            if part
        ]
        if event.confidence is not None:
            context_parts.append(f"{event.confidence * 100:.0f}% confidence")
    else:
        title = "Observed site visit"
        context_parts = (
            [_escaped(event.camera_source)] if event.camera_source else []
        )
    context = " · ".join(context_parts) or "No location label"
    return (
        '<article class="row"><div class="row-top"><div><p class="row-title">'
        + title
        + '</p><p class="row-meta">'
        + _escaped(_format_timestamp(event.started_at))
        + " · "
        + context
        + '</p></div><span class="duration">'
        + _escaped(_format_duration(event.duration_seconds))
        + "</span></div></article>"
    )


def _portal_payload(
    summary: PortalSummary,
    timeline: tuple[PortalTimelineEvent, ...],
) -> dict[str, object]:
    """Build an explicit public shape that omits IDs and filesystem paths."""

    return {
        "profile": {
            "full_name": summary.full_name,
            "photo_url": (
                "/portal/photo" if summary.profile_photo is not None else None
            ),
            "photo_captured_at": (
                summary.profile_photo.captured_at
                if summary.profile_photo is not None
                else None
            ),
        },
        "activity": {
            "label": "Observed activity",
            "interaction_definition": "Estimated visual proximity",
            "total_presence_seconds": summary.total_presence_seconds,
            "total_interaction_seconds": summary.total_interaction_seconds,
            "first_entry_at": summary.first_entry_at,
            "latest_exit_at": summary.latest_exit_at,
        },
        "visits": [
            {
                "entered_at": visit.entered_at,
                "exited_at": visit.exited_at,
                "duration_seconds": visit.duration_seconds,
                "status": (
                    "IN_PROGRESS" if visit.exited_at is None else "COMPLETED"
                ),
                "entry_camera_source": visit.entry_camera_source,
                "exit_camera_source": visit.exit_camera_source,
            }
            for visit in summary.visits
        ],
        "interactions": [
            {
                "counterpart_name": item.counterpart_full_name,
                "total_duration_seconds": item.total_duration_seconds,
                "interaction_count": item.interaction_count,
                "last_interaction_at": item.last_interaction_at,
            }
            for item in summary.interactions
        ],
        "timeline": [
            {
                "event_type": event.event_type,
                "started_at": event.started_at,
                "ended_at": event.ended_at,
                "duration_seconds": event.duration_seconds,
                "counterpart_name": event.counterpart_full_name,
                "camera_source": event.camera_source,
                "zone_label": event.zone_label,
                "confidence": event.confidence,
            }
            for event in timeline
        ],
    }


def _simple_page(title: str, message: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="referrer" content="no-referrer"><title>'
        + _escaped(title)
        + " · CCTV Portal</title><style>"
        + _PAGE_STYLE
        + "</style></head><body>"
        '<main class="error-card"><p class="eyebrow">Personal activity portal</p>'
        "<h1>"
        + _escaped(title)
        + "</h1><p>"
        + _escaped(message)
        + "</p></main></body></html>"
    )
