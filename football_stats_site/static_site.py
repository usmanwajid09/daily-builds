"""A tiny, dependency-free static file server for the frontend.

No third-party framework (Flask/Werkzeug's static handling, etc.) --
matches the rest of the repo's stdlib-only WSGI convention from app.py.
Serves whatever is in a given directory (the frontend's HTML/CSS/JS),
with basic path-traversal protection and an SPA-style fallback to
index.html for unknown non-asset paths (the frontend does client-side
tab switching in app.js rather than real server routes, so a future
deep link like ``/team/River%20Athletic`` should still load the app
shell instead of 404ing).
"""
from __future__ import annotations

import json
import os
from typing import Callable

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
}
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


def _json_error(status: str, message: str) -> tuple[str, list[tuple[str, str]], bytes]:
    # Use json.dumps rather than hand-rolling the JSON string: an
    # earlier version built this with repr()+str.replace("'", '"'),
    # which produced invalid JSON whenever `message` contained an
    # apostrophe (e.g. a 404 for a path like "/don't-exist.js" -- the
    # apostrophe in the echoed path got turned into a stray unescaped
    # double quote, breaking the response for any JSON client).
    body = json.dumps({"error": message}).encode("utf-8")
    headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
    return status, headers, body


class StaticFileApp:
    """Serves static files from ``root_dir``."""

    def __init__(self, root_dir: str, index_file: str = "index.html"):
        self.root_dir = os.path.realpath(root_dir)
        self.index_file = index_file

    def _resolve(self, url_path: str) -> str | None:
        """Map a URL path onto a real file inside root_dir, or None if
        there's no such file (or the path tries to escape root_dir).
        """
        rel = url_path.lstrip("/") or self.index_file
        candidate = os.path.realpath(os.path.join(self.root_dir, rel))
        root_with_sep = self.root_dir + os.sep
        if not (candidate == self.root_dir or candidate.startswith(root_with_sep)):
            return None  # path traversal attempt (e.g. "../../etc/passwd")
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, self.index_file)
        return candidate if os.path.isfile(candidate) else None

    def __call__(self, environ: dict, start_response: Callable) -> list[bytes]:
        method = environ.get("REQUEST_METHOD", "GET")
        url_path = environ.get("PATH_INFO", "/")

        if method != "GET":
            status, headers, body = _json_error("405 Method Not Allowed", "method not allowed")
            start_response(status, headers)
            return [body]

        resolved = self._resolve(url_path)
        looks_like_asset_request = "." in os.path.basename(url_path)

        if resolved is None:
            if looks_like_asset_request:
                status, headers, body = _json_error("404 Not Found", f"not found: {url_path}")
                start_response(status, headers)
                return [body]
            # SPA fallback for path-like routes with no extension.
            resolved = self._resolve("/")
            if resolved is None:
                status, headers, body = _json_error(
                    "404 Not Found", "static site index.html is missing"
                )
                start_response(status, headers)
                return [body]

        ext = os.path.splitext(resolved)[1]
        content_type = _CONTENT_TYPES.get(ext, _DEFAULT_CONTENT_TYPE)
        with open(resolved, "rb") as f:
            body = f.read()
        start_response("200 OK", [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
        ])
        return [body]
