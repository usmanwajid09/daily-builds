import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from football_stats_site.app import create_app
from football_stats_site.data import generate_season
from football_stats_site.tests.wsgi_client import get, post


def make_test_app():
    # Small, fast, fully deterministic season for API tests.
    return create_app(generate_season(cutoff_matchday=3))


def test_health():
    app = make_test_app()
    resp = get(app, "/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_teams_lists_all_ten():
    app = make_test_app()
    resp = get(app, "/api/teams")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["teams"]) == 10
    assert {"name": "River Athletic", "short_code": "RVR"} in body["teams"]


def test_standings_returns_sorted_table():
    app = make_test_app()
    resp = get(app, "/api/standings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["standings"], "standings should not be empty"
    points = [row["points"] for row in body["standings"]]
    assert points == sorted(points, reverse=True)


def test_matches_default_returns_all():
    app = make_test_app()
    season = generate_season(cutoff_matchday=3)
    resp = get(app, "/api/matches")
    assert resp.status_code == 200
    assert resp.json()["count"] == len(season.matches)


def test_fixtures_only_unplayed():
    app = make_test_app()
    resp = get(app, "/api/fixtures")
    body = resp.json()
    assert body["count"] > 0
    assert all(f["status"] == "SCHEDULED" for f in body["fixtures"])


def test_results_only_played():
    app = make_test_app()
    resp = get(app, "/api/results")
    body = resp.json()
    assert body["count"] > 0
    assert all(r["status"] == "FT" for r in body["results"])


def test_matches_filtered_by_matchday():
    app = make_test_app()
    resp = get(app, "/api/matches?matchday=1")
    body = resp.json()
    assert body["count"] == 5  # 10 teams -> 5 matches per matchday
    assert all(m["matchday"] == 1 for m in body["matches"])


def test_matches_filtered_by_team():
    app = make_test_app()
    resp = get(app, "/api/matches?team=River Athletic")
    body = resp.json()
    assert body["count"] > 0
    assert all(
        "River Athletic" in (m["home_team"], m["away_team"]) for m in body["matches"]
    )


def test_matches_invalid_matchday_is_400():
    app = make_test_app()
    resp = get(app, "/api/matches?matchday=notanumber")
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_matches_unknown_team_is_404():
    app = make_test_app()
    resp = get(app, "/api/matches?team=Nonexistent FC")
    assert resp.status_code == 404
    assert "error" in resp.json()


def test_unknown_route_is_404():
    app = make_test_app()
    resp = get(app, "/api/nope")
    assert resp.status_code == 404


def test_write_method_is_405():
    app = make_test_app()
    resp = post(app, "/api/standings")
    assert resp.status_code == 405


def test_write_method_on_unknown_route_is_404_not_405():
    # Route existence should be checked before method: a POST to a path
    # that was never a valid route is a 404, not a misleading "405 method
    # not allowed on this route" (the route was never allowed at all).
    app = make_test_app()
    resp = post(app, "/api/nope")
    assert resp.status_code == 404


def test_response_content_type_is_json():
    app = make_test_app()
    resp = get(app, "/api/health")
    assert resp.headers["Content-Type"] == "application/json"


def test_top_scorers_default_and_limit():
    app = make_test_app()
    resp = get(app, "/api/top-scorers")
    assert resp.status_code == 200
    all_scorers = resp.json()["top_scorers"]

    resp2 = get(app, "/api/top-scorers?limit=2")
    assert resp2.status_code == 200
    limited = resp2.json()["top_scorers"]
    assert limited == all_scorers[:2]


def test_top_scorers_invalid_limit_is_400():
    app = make_test_app()
    resp = get(app, "/api/top-scorers?limit=abc")
    assert resp.status_code == 400
    resp2 = get(app, "/api/top-scorers?limit=0")
    assert resp2.status_code == 400


def test_players_lists_everyone_and_filters_by_team():
    app = make_test_app()
    resp = get(app, "/api/players")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 160  # 10 teams x 16-player rosters

    resp2 = get(app, "/api/players?team=River Athletic")
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["count"] == 16
    assert all(p["team"] == "River Athletic" for p in body2["players"])


def test_players_unknown_team_is_404():
    app = make_test_app()
    resp = get(app, "/api/players?team=Nonexistent FC")
    assert resp.status_code == 404


def test_players_search_is_case_insensitive_substring():
    app = make_test_app()
    resp = get(app, "/api/players?q=KWAME")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0
    assert all("kwame" in p["name"].lower() for p in body["players"])


def test_search_finds_teams_and_players():
    app = make_test_app()
    resp = get(app, "/api/search?q=river")
    assert resp.status_code == 200
    body = resp.json()
    assert any(t["name"] == "River Athletic" for t in body["teams"])


def test_search_requires_nonempty_q():
    app = make_test_app()
    resp = get(app, "/api/search")
    assert resp.status_code == 400
    resp2 = get(app, "/api/search?q=")
    assert resp2.status_code == 400


def test_live_returns_matches_and_advances_each_call():
    app = make_test_app()
    resp1 = get(app, "/api/live")
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["matches"]

    resp2 = get(app, "/api/live")
    body2 = resp2.json()
    # Each GET ticks the shared simulator, so the clock should have
    # moved forward between the two polls.
    assert body2["clock_minute"] > body1["clock_minute"]


def test_live_degrades_gracefully_when_no_unplayed_matchday_exists():
    # A season where every matchday is already played has nothing left
    # to simulate as "live" -- the app must still construct and the
    # endpoint must still return 200, not crash.
    from football_stats_site.data import round_robin_schedule

    season = generate_season(cutoff_matchday=len(round_robin_schedule(
        [t.name for t in generate_season().teams]
    )))
    app = create_app(season)
    resp = get(app, "/api/live")
    assert resp.status_code == 200
    assert resp.json()["matches"] == []


def test_team_detail_returns_roster_standing_and_form():
    app = make_test_app()
    resp = get(app, "/api/teams/River Athletic")
    assert resp.status_code == 200
    body = resp.json()
    assert body["team"]["name"] == "River Athletic"
    assert len(body["roster"]) == 16
    assert body["standing"]["team"] == "River Athletic"
    assert isinstance(body["recent_form"], list)
    assert isinstance(body["upcoming_fixtures"], list)


def test_team_detail_unknown_team_is_404():
    app = make_test_app()
    resp = get(app, "/api/teams/Nonexistent FC")
    assert resp.status_code == 404
    assert "known_teams" in resp.json()


def test_team_detail_write_method_is_405_not_404():
    app = make_test_app()
    resp = post(app, "/api/teams/River Athletic")
    assert resp.status_code == 405


def test_team_detail_bare_prefix_with_no_name_is_404():
    # "/api/teams/" with nothing after it is not a valid team-detail
    # route (empty team name) -- it should 404, not crash trying to
    # look up a team called "".
    app = make_test_app()
    resp = get(app, "/api/teams/")
    assert resp.status_code == 404
