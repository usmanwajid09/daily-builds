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
from urllib.parse import parse_qs, unquote

from .data import generate_season
from .live import LiveScoreSimulator
from .models import Match, Season
from .standings import compute_standings, top_scorers

JsonResponse = tuple[int, dict]


class FootballStatsApp:
    """WSGI app wrapping a single in-memory Season.

    The season is generated once at construction time (or injected, e.g.
    by tests) and served read-only -- there are no write endpoints in
    this milestone.
    """

    def __init__(self, season: Season | None = None, live_sim: "LiveScoreSimulator | None | bool" = None):
        self.season = season if season is not None else generate_season()

        # live_sim: None (default) -> try to build one automatically from
        # the season's first unplayed matchday; False -> explicitly
        # disable (used by tests that don't care about /api/live);
        # an actual LiveScoreSimulator instance -> use it as-is (lets
        # tests control the clock directly).
        if live_sim is False:
            self.live_sim: LiveScoreSimulator | None = None
        elif live_sim is None:
            try:
                self.live_sim = LiveScoreSimulator(self.season)
            except ValueError:
                # No unplayed matchday to simulate (e.g. a fully-played
                # test season) -- /api/live degrades gracefully instead
                # of the app failing to construct at all.
                self.live_sim = None
        else:
            self.live_sim = live_sim

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

    def _handle_top_scorers(self, query: dict) -> JsonResponse:
        limit = None
        if "limit" in query:
            raw = query["limit"][0]
            try:
                limit = int(raw)
            except ValueError:
                return 400, {"error": f"limit must be an integer, got {raw!r}"}
            if limit < 1:
                return 400, {"error": f"limit must be >= 1, got {limit}"}
        return 200, {"top_scorers": top_scorers(self.season.matches, limit=limit)}

    def _all_players(self) -> list[dict]:
        players = []
        for team in self.season.teams:
            for p in team.roster:
                players.append({
                    "name": p.name,
                    "position": p.position,
                    "squad_number": p.squad_number,
                    "team": team.name,
                })
        return players

    def _handle_players(self, query: dict) -> JsonResponse:
        players = self._all_players()

        if "team" in query:
            team = query["team"][0]
            valid_teams = set(self.season.team_names())
            if team not in valid_teams:
                return 404, {"error": f"unknown team {team!r}", "known_teams": sorted(valid_teams)}
            players = [p for p in players if p["team"] == team]

        for key in ("q", "search"):
            if key in query:
                needle = query[key][0].strip().lower()
                players = [p for p in players if needle in p["name"].lower()]
                break

        return 200, {"count": len(players), "players": players}

    def _handle_search(self, query: dict) -> JsonResponse:
        if "q" not in query or not query["q"][0].strip():
            return 400, {"error": "search requires a non-empty ?q= parameter"}
        needle = query["q"][0].strip().lower()

        teams = [
            {"name": t.name, "short_code": t.short_code}
            for t in self.season.teams
            if needle in t.name.lower() or needle in t.short_code.lower()
        ]
        players = [p for p in self._all_players() if needle in p["name"].lower()]

        return 200, {"query": query["q"][0], "teams": teams, "players": players}

    def _handle_live(self, query: dict) -> JsonResponse:
        if self.live_sim is None:
            return 200, {
                "matchday": None,
                "clock_minute": 0,
                "finished": True,
                "matches": [],
                "note": "no unplayed matchday available to simulate as live",
            }
        # Advancing the clock on every poll is what makes this feel
        # "live" -- see live.py's module docstring for the reasoning.
        self.live_sim.tick()
        return 200, self.live_sim.snapshot()

    def _handle_team_detail(self, team_name: str, query: dict) -> JsonResponse:
        team = self.season.find_team(team_name)
        if team is None:
            return 404, {
                "error": f"unknown team {team_name!r}",
                "known_teams": sorted(self.season.team_names()),
            }

        table = compute_standings(self.season.matches, self.season.team_names())
        record = next((row for row in table if row["team"] == team_name), None)

        team_matches = [
            m for m in self.season.matches
            if team_name in (m.home_team, m.away_team)
        ]
        played = sorted(
            (m for m in team_matches if m.is_played),
            key=lambda m: (m.matchday, m.id),
        )
        upcoming = sorted(
            (m for m in team_matches if not m.is_played),
            key=lambda m: (m.matchday, m.id),
        )

        recent_form = []
        for m in played[-5:][::-1]:
            is_home = m.home_team == team_name
            gf = m.home_score if is_home else m.away_score
            ga = m.away_score if is_home else m.home_score
            result = "W" if gf > ga else ("L" if gf < ga else "D")
            recent_form.append({"match": m.to_dict(), "result": result})

        return 200, {
            "team": {"name": team.name, "short_code": team.short_code},
            "roster": [
                {"name": p.name, "position": p.position, "squad_number": p.squad_number}
                for p in team.roster
            ],
            "standing": record,
            "recent_form": recent_form,
            "upcoming_fixtures": [m.to_dict() for m in upcoming[:5]],
        }

    ROUTES: dict[str, str] = {
        "/api/health": "_handle_health",
        "/api/teams": "_handle_teams",
        "/api/standings": "_handle_standings",
        "/api/matches": "_handle_matches",
        "/api/fixtures": "_handle_fixtures",
        "/api/results": "_handle_results",
        "/api/top-scorers": "_handle_top_scorers",
        "/api/players": "_handle_players",
        "/api/search": "_handle_search",
        "/api/live": "_handle_live",
    }

    TEAM_DETAIL_PREFIX = "/api/teams/"

    # -- WSGI entry point -------------------------------------------------

    def __call__(self, environ: dict, start_response: Callable) -> list[bytes]:
        path = environ.get("PATH_INFO", "/")
        query = parse_qs(environ.get("QUERY_STRING", ""))
        method = environ.get("REQUEST_METHOD", "GET")

        # Resolve which handler (if any) this path maps to -- either an
        # exact static route, or the dynamic /api/teams/<name> pattern --
        # *before* looking at the method, so a POST to a nonexistent path
        # reports 404 (route not found) rather than a misleading 405
        # (method not allowed on a route that was never valid anyway).
        handler_name = self.ROUTES.get(path)
        team_name = None
        if handler_name is None and path.startswith(self.TEAM_DETAIL_PREFIX):
            remainder = path[len(self.TEAM_DETAIL_PREFIX):]
            if remainder:
                team_name = unquote(remainder)
                handler_name = "_handle_team_detail"

        if handler_name is None:
            status, body = 404, {"error": f"no such route: {path}"}
        elif method != "GET":
            status, body = 405, {"error": f"method {method} not allowed, this API is read-only"}
        elif team_name is not None:
            status, body = self._handle_team_detail(team_name, query)
        else:
            handler = getattr(self, handler_name)
            status, body = handler(query)

        payload = json.dumps(body).encode("utf-8")
        status_line = self._STATUS_LINES.get(status, f"{status} Error")
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(payload))),
        ]
        start_response(status_line, headers)
        return [payload]

    _STATUS_LINES = {
        200: "200 OK",
        400: "400 Bad Request",
        404: "404 Not Found",
        405: "405 Method Not Allowed",
    }


def create_app(season: Season | None = None) -> FootballStatsApp:
    return FootballStatsApp(season=season)
