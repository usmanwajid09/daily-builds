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
