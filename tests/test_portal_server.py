"""Focused integration tests for the dependency-free demo portal."""

from __future__ import annotations

import json
import threading
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from http.cookies import SimpleCookie
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

from src.activity_repository import ActivityRepository
from src.database import database_connection
from src.portal_server import (
    SESSION_COOKIE_NAME,
    create_portal_server,
)


PEOPLE_SCHEMA = """
CREATE TABLE people (
    person_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    person_type TEXT NOT NULL
);
"""

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
)


class AdjustableClock:
    def __init__(self, value: float = 10_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass(frozen=True)
class HTTPResult:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes

    def header(self, name: str) -> str | None:
        expected = name.casefold()
        for header_name, value in self.headers:
            if header_name.casefold() == expected:
                return value
        return None


class PortalServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database_path = self.root / "activity.db"
        self.photo_path = self.root / "private-profile.png"
        self.photo_path.write_bytes(TINY_PNG)
        self.now = datetime.now(timezone.utc).replace(microsecond=0)

        with database_connection(self.database_path) as connection:
            connection.executescript(PEOPLE_SCHEMA)
            connection.executemany(
                """
                INSERT INTO people (person_id, full_name, person_type)
                VALUES (?, ?, ?);
                """,
                [
                    (
                        "person-a",
                        'Alice <script>alert("x")</script>',
                        "VISITOR",
                    ),
                    ("person-b", "Bob & Sons", "AI_TEAM"),
                ],
            )

        self.repository = ActivityRepository(self.database_path)
        self.repository.set_profile_photo(
            "person-a",
            str(self.photo_path),
            captured_at=self.now - timedelta(hours=4),
        )
        visit = self.repository.start_visit(
            "person-a",
            self.now - timedelta(hours=3),
            entry_camera_source="Lobby <East>",
        )
        self.repository.end_visit(
            visit,
            self.now - timedelta(hours=2),
            exit_camera_source="Lobby West",
        )
        self.repository.record_proximity_interaction(
            "person-a",
            "person-b",
            self.now - timedelta(minutes=90),
            self.now - timedelta(minutes=75),
            confidence=0.91,
            camera_source="Room 1",
            zone_label="Meeting & table",
        )
        self.credential = self.repository.issue_portal_credential(
            "person-a",
            expires_at=self.now + timedelta(hours=1),
            issued_at=self.now - timedelta(minutes=1),
        )

        self.clock = AdjustableClock()
        self.server = create_portal_server(
            ("127.0.0.1", 0),
            self.repository,
            session_ttl_seconds=5,
            photo_root=self.root,
            clock=self.clock,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        cookie: str | None = None,
    ) -> HTTPResult:
        host, port = self.server.server_address[:2]
        connection = HTTPConnection(host, port, timeout=3)
        headers = {"Cookie": cookie} if cookie is not None else {}
        try:
            connection.request(method, path, headers=headers)
            response = connection.getresponse()
            return HTTPResult(
                status=response.status,
                headers=tuple(response.getheaders()),
                body=response.read(),
            )
        finally:
            connection.close()

    @staticmethod
    def session_cookie(result: HTTPResult) -> tuple[str, str]:
        raw_cookie = result.header("Set-Cookie")
        if raw_cookie is None:
            raise AssertionError("Response did not set a session cookie.")
        jar = SimpleCookie()
        jar.load(raw_cookie)
        morsel = jar[SESSION_COOKIE_NAME]
        return f"{SESSION_COOKIE_NAME}={morsel.value}", raw_cookie

    def exchange(self, token: str | None = None) -> tuple[str, HTTPResult]:
        credential = token or self.credential.token
        result = self.request("/q/" + quote(credential, safe=""))
        cookie, _ = self.session_cookie(result)
        return cookie, result

    def test_qr_exchange_dashboard_api_and_photo(self) -> None:
        unauthenticated_photo = self.request("/portal/photo")
        self.assertEqual(unauthenticated_photo.status, 401)

        cookie, exchange = self.exchange()
        raw_cookie = exchange.header("Set-Cookie") or ""
        self.assertEqual(exchange.status, 303)
        self.assertEqual(exchange.header("Location"), "/portal")
        self.assertIn("HttpOnly", raw_cookie)
        self.assertIn("SameSite=Strict", raw_cookie)
        self.assertIn("Max-Age=5", raw_cookie)
        self.assertNotIn(self.credential.token, raw_cookie)
        self.assertNotIn("person-a", raw_cookie)
        self.assertEqual(exchange.body, b"")
        self.assertEqual(exchange.header("Cache-Control"), "no-store, max-age=0")
        self.assertEqual(exchange.header("Referrer-Policy"), "no-referrer")

        dashboard = self.request("/portal", cookie=cookie)
        dashboard_text = dashboard.body.decode("utf-8")
        self.assertEqual(dashboard.status, 200)
        self.assertIn("Observed presence", dashboard_text)
        self.assertIn("Estimated interaction time", dashboard_text)
        self.assertIn(
            "Alice &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;",
            dashboard_text,
        )
        self.assertNotIn('<script>alert("x")</script>', dashboard_text)
        self.assertIn("Bob &amp; Sons", dashboard_text)
        self.assertIn("Lobby &lt;East&gt;", dashboard_text)
        self.assertNotIn("https://", dashboard_text)
        self.assertNotIn("http://", dashboard_text)
        self.assertIn("default-src 'none'", dashboard.header("Content-Security-Policy") or "")
        self.assertEqual(dashboard.header("X-Content-Type-Options"), "nosniff")
        self.assertEqual(dashboard.header("X-Frame-Options"), "DENY")

        api = self.request("/api/v1/portal/me", cookie=cookie)
        api_text = api.body.decode("utf-8")
        payload = json.loads(api_text)
        self.assertEqual(api.status, 200)
        self.assertEqual(payload["profile"]["photo_url"], "/portal/photo")
        self.assertEqual(payload["interactions"][0]["counterpart_name"], "Bob & Sons")
        self.assertNotIn(str(self.photo_path), api_text)
        self.assertNotIn("profile_photo_path", api_text)
        self.assertNotIn("person_id", api_text)
        self.assertNotIn("person-a", api_text)

        photo = self.request("/portal/photo", cookie=cookie)
        self.assertEqual(photo.status, 200)
        self.assertEqual(photo.header("Content-Type"), "image/png")
        self.assertEqual(photo.header("Cache-Control"), "no-store, max-age=0")
        self.assertEqual(photo.body, TINY_PNG)

    def test_bad_expired_and_revoked_credentials_share_generic_404(self) -> None:
        expired = self.repository.issue_portal_credential(
            "person-a",
            expires_at=self.now - timedelta(hours=1),
            issued_at=self.now - timedelta(hours=2),
        )
        revoked = self.repository.issue_portal_credential(
            "person-a",
            expires_at=self.now + timedelta(hours=1),
            issued_at=self.now - timedelta(minutes=2),
        )
        self.repository.revoke_portal_credential(
            revoked.credential_id,
            revoked_at=self.now - timedelta(minutes=1),
        )

        results = [
            self.request("/q/" + "A" * 43),
            self.request("/q/" + quote(expired.token, safe="")),
            self.request("/q/" + quote(revoked.token, safe="")),
        ]

        for result in results:
            self.assertEqual(result.status, 404)
            self.assertEqual(result.header("Cache-Control"), "no-store, max-age=0")
            self.assertNotIn("expired", result.body.decode("utf-8").casefold())
            self.assertNotIn("revoked", result.body.decode("utf-8").casefold())
        self.assertEqual(results[0].body, results[1].body)
        self.assertEqual(results[1].body, results[2].body)

    def test_session_expiry_logout_and_health(self) -> None:
        health = self.request("/health")
        self.assertEqual(health.status, 200)
        self.assertEqual(json.loads(health.body), {"status": "ok"})

        cookie, _ = self.exchange()
        self.clock.advance(6)
        expired_api = self.request("/api/v1/portal/me", cookie=cookie)
        self.assertEqual(expired_api.status, 401)
        self.assertEqual(
            json.loads(expired_api.body),
            {"error": "authentication_required"},
        )
        self.assertIn("Max-Age=0", expired_api.header("Set-Cookie") or "")

        new_cookie, _ = self.exchange()
        self.assertNotEqual(cookie, new_cookie)
        get_logout = self.request("/logout", cookie=new_cookie)
        self.assertEqual(get_logout.status, 405)
        self.assertEqual(get_logout.header("Allow"), "POST")
        still_authenticated = self.request("/portal", cookie=new_cookie)
        self.assertEqual(still_authenticated.status, 200)

        logout = self.request(
            "/logout",
            method="POST",
            cookie=new_cookie,
        )
        self.assertEqual(logout.status, 200)
        self.assertIn("Max-Age=0", logout.header("Set-Cookie") or "")
        after_logout = self.request("/portal", cookie=new_cookie)
        self.assertEqual(after_logout.status, 401)

    def test_profile_photo_cannot_escape_configured_root(self) -> None:
        with TemporaryDirectory() as outside_directory:
            outside_photo = Path(outside_directory) / "outside.png"
            outside_photo.write_bytes(TINY_PNG)
            self.repository.set_profile_photo(
                "person-a",
                str(outside_photo),
                captured_at=self.now,
            )
            cookie, _ = self.exchange()
            result = self.request("/portal/photo", cookie=cookie)

        self.assertEqual(result.status, 404)
        self.assertNotIn(
            str(outside_photo),
            result.body.decode("utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
