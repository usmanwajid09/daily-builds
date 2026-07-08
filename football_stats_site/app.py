"""A tiny, dependency-free JSON API over the generated season.

Deliberately implemented as a plain WSGI application using only the
standard library (no Flask/FastAPI): this sandbox has no working
internet access to `pip install` new packages reliably (see data.py for
the same issue affecting real data ingestion), and the repo's existing
convention (numpy-only GPT, from-scratch trading indicators) favors
minimal external dependencies anyway.

Routes:
    GET /api/health              -> {"status": "ok"}
    GET /api/teams                -> [{"name", "short_code"}, ...]
    GET /api/standings             -> league table, see standings.py
    GET /api/matches   [?matchday=][?team=]  -> all matches
    GET /api/fixtures  [?matchday=][?team=]  -> unplayed matches only
    GET /api/results   [?matchday=][?team=]  -> played matches only

Any other path returns 404; a non-integer ``matchday`` returns 400.
"""
from __future__ import annotations

import json
from typing import Callable
from urllib.parse import parse_qs

from .data import generate_season
from .models import Match, Season
from .standings import compute_standings

JsonResponse = tuple[int, dict]


class FootballStatsApp:
    """WSGI app wrapping a single in-memory Season.

    The season is generated once at construction time (or injected, e.g.
    by tests) and served read-only -- there are no write endpoints in
    this milestone.
    """

    def __init__(self, season: Season | None = None):
        self.season = season if season is not None else generate_season()

    # -- route handlers -------------------------------------------------

    def _handle_health(self, query: dict) -> JsonResponse:
        return 200, {"status": "ok"}

    def _handle_teams(self, query: dict) -> JsonResponse:
        teams = [
            {"name": t.name, "short_code": t.short_code} for t in self.season.teams
        ]
        return 200, {"teams": teams}

    def _handle_standings(self, query: dict) -> JsonResponse:
        table = compute_standings(self.season.matches, self.season.team_names())
        return 200, {"season": self.season.name, "standings": table}

    def _filtered_matches(self, query: dict) -> tuple[list[Match], JsonResponse | None]:
        """Shared query-param filtering for matches/fixtures/results.

        Returns ``(matches, None)`` on success or ``([], error_response)``
        if a query parameter was invalid.
        """
        matches = self.season.matches

        if "matchday" in query:
            raw = query["matchday"][0]
            try:
                matchday = int(raw)
            except ValueError:
                return [], (400, {"error": f"matchday must be an integer, got {raw!r}"})
            matches = [m for m in matches if m.matchday == matchday]

        if "team" in query:
            team = query["team"][0]
            valid_teams = set(self.season.team_names())
            if team not in valid_teams:
                return [], (
                    404,
                    {"error": f"unknown team {team!r}", "known_teams": sorted(valid_teams)},
                )
            matches = [m for m in matches if team in (m.home_team, m.away_team)]

        return matches, None

    def _handle_matches(self, query: dict) -> JsonResponse:
        matches, error = self._filtered_matches(query)
        if error is not None:
            return error
        return 200, {"count": len(matches), "matches": [m.to_dict() for m in matches]}

    def _handle_fixtures(self, query: dict) -> JsonResponse:
        matches, error = self._filtered_matches(query)
        if error is not None:
            return error
        fixtures = [m for m in matches if not m.is_played]
        return 200, {"count": len(fixtures), "fixtures": [m.to_dict() for m in fixtures]}

    def _handle_results(self, query: dict) -> JsonResponse:
        matches, error = self._filtered_matches(query)
        if error is not None:
            return error
        results = [m for m in matches if m.is_played]
        return 200, {"count": len(results), "results": [m.to_dict() for m in results]}

    ROUTES: dict[str, str] = {
        "/api/health": "_handle_health",
        "/api/teams": "_handle_teams",
        "/api/standings": "_handle_standings",
        "/api/matches": "_handle_matches",
        "/api/fixtures": "_handle_fixtures",
        "/api/results": "_handle_results",
    }

    # -- WSGI entry point -------------------------------------------------

    def __call__(self, environ: dict, start_response: Callable) -> list[bytes]:
        path = environ.get("PATH_INFO", "/")
        query = parse_qs(environ.get("QUERY_STRING", ""))
        method = environ.get("REQUEST_METHOD", "GET")

        if method != "GET":
            status, body = 405, {"error": f"method {method} not allowed, this API is read-only"}
        else:
            handler_name = self.ROUTES.get(path)
            if handler_name is None:
                status, body = 404, {"error": f"no such route: {path}"}
            else:
                handler = getattr(self, handler_name)
                status, body = handler(query)

        payload = json.dumps(body).encode("utf-8")
        status_line = {
            200: "200 OK",
            400: "400 Bad Request",
            404: "404 Not Found",
            405: "405 Method Not Allowed",
        }[status]
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(payload))),
        ]
        start_response(status_line, headers)
        return [payload]


def create_app(season: Season | None = None) -> FootballStatsApp:
    return FootballStatsApp(season=season)
