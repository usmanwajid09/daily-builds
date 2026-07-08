"""Combines the JSON API (app.py) with the static frontend (static_site.py)
into a single WSGI app.

Requests to ``/api/*`` are delegated to ``FootballStatsApp``; everything
else is served as a static file from the ``static/`` directory next to
this module (the frontend's index.html/styles.css/app.js).
"""
from __future__ import annotations

import os
from typing import Callable

from .app import FootballStatsApp, create_app
from .models import Season
from .static_site import StaticFileApp

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class CombinedApp:
    def __init__(
        self,
        api_app: FootballStatsApp | None = None,
        static_dir: str = STATIC_DIR,
    ):
        self.api_app = api_app if api_app is not None else create_app()
        self.static_app = StaticFileApp(static_dir)

    def __call__(self, environ: dict, start_response: Callable) -> list[bytes]:
        path = environ.get("PATH_INFO", "/")
        if path.startswith("/api/"):
            return self.api_app(environ, start_response)
        return self.static_app(environ, start_response)


def create_combined_app(
    season: Season | None = None, static_dir: str = STATIC_DIR
) -> CombinedApp:
    return CombinedApp(api_app=create_app(season), static_dir=static_dir)
