"""A minimal, dependency-free WSGI test client.

Calls the app object directly by constructing a WSGI ``environ`` dict --
no real socket or thread needed, so tests are fast and fully synchronous.
"""
from __future__ import annotations

import io
import json
from typing import Callable


class WSGIResponse:
    def __init__(self, status: str, headers: list[tuple[str, str]], body: bytes):
        self.status_line = status
        self.status_code = int(status.split(" ", 1)[0])
        self.headers = dict(headers)
        self.body = body

    def json(self):
        return json.loads(self.body.decode("utf-8"))


def get(app: Callable, path: str) -> WSGIResponse:
    """Issue a GET request against a WSGI app. ``path`` may include a query
    string, e.g. "/api/matches?matchday=1".
    """
    if "?" in path:
        path_info, query_string = path.split("?", 1)
    else:
        path_info, query_string = path, ""

    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": path_info,
        "QUERY_STRING": query_string,
        "SERVER_NAME": "testserver",
        "SERVER_PORT": "80",
        "wsgi.input": io.BytesIO(b""),
        "wsgi.errors": io.StringIO(),
        "wsgi.version": (1, 0),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
        "wsgi.url_scheme": "http",
    }

    captured: dict = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = headers

    body_chunks = app(environ, start_response)
    body = b"".join(body_chunks)
    return WSGIResponse(captured["status"], captured["headers"], body)


def post(app: Callable, path: str) -> WSGIResponse:
    """Issue a POST request (used only to test the 405 read-only guard)."""
    if "?" in path:
        path_info, query_string = path.split("?", 1)
    else:
        path_info, query_string = path, ""

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": path_info,
        "QUERY_STRING": query_string,
        "SERVER_NAME": "testserver",
        "SERVER_PORT": "80",
        "wsgi.input": io.BytesIO(b""),
        "wsgi.errors": io.StringIO(),
        "wsgi.version": (1, 0),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
        "wsgi.url_scheme": "http",
    }

    captured: dict = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = headers

    body_chunks = app(environ, start_response)
    body = b"".join(body_chunks)
    return WSGIResponse(captured["status"], captured["headers"], body)
