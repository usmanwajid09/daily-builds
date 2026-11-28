"""Tests for the mood_tracker_api Flask app, using Flask's test client."""
from datetime import date, timedelta

import pytest

from mood_tracker_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _iso(days_ago=0):
    return (date.today() - timedelta(days=days_ago)).isoformat()


def _add(client, days_ago=0, mood=4, note="ok", tags=None):
    payload = {"date": _iso(days_ago), "mood": mood}
    if note is not None:
        payload["note"] = note
    if tags is not None:
        payload["tags"] = tags
    return client.post("/entries", json=payload)


def test_list_starts_empty(client):
    assert client.get("/entries").get_json() == []


def test_create_entry_returns_201(client):
    resp = _add(client, days_ago=1, mood=4, note="Good day", tags=["work"])
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["date"] == _iso(1)
    assert body["mood"] == 4
    assert body["note"] == "Good day"
    assert body["tags"] == ["work"]


def test_create_defaults_note_and_tags(client):
    resp = client.post("/entries", json={"date": _iso(0), "mood": 3})
    assert resp.status_code == 201
    assert resp.get_json()["note"] == ""
    assert resp.get_json()["tags"] == []


def test_duplicate_date_returns_409(client):
    assert _add(client, days_ago=0, mood=3).status_code == 201
    resp = _add(client, days_ago=0, mood=5)
    assert resp.status_code == 409
    assert "error" in resp.get_json()


@pytest.mark.parametrize("mood", [0, 6, -1])
def test_invalid_mood_out_of_range_returns_400(client, mood):
    resp = client.post("/entries", json={"date": _iso(0), "mood": mood})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


@pytest.mark.parametrize("mood", ["happy", 3.5, True, None])
def test_non_integer_mood_returns_400(client, mood):
    resp = client.post("/entries", json={"date": _iso(0), "mood": mood})
    assert resp.status_code == 400


def test_missing_mood_returns_400(client):
    resp = client.post("/entries", json={"date": _iso(0)})
    assert resp.status_code == 400


@pytest.mark.parametrize("bad_date", ["not-a-date", "2026-13-40", "2026-02-30", "28-11-2026", 20261128])
def test_bad_date_returns_400(client, bad_date):
    resp = client.post("/entries", json={"date": bad_date, "mood": 3})
    assert resp.status_code == 400


def test_missing_date_returns_400(client):
    resp = client.post("/entries", json={"mood": 3})
    assert resp.status_code == 400


def test_note_not_string_returns_400(client):
    resp = client.post("/entries", json={"date": _iso(0), "mood": 3, "note": 42})
    assert resp.status_code == 400


def test_tags_not_list_of_strings_returns_400(client):
    resp = client.post("/entries", json={"date": _iso(0), "mood": 3, "tags": "work"})
    assert resp.status_code == 400
    resp = client.post("/entries", json={"date": _iso(0), "mood": 3, "tags": ["work", 7]})
    assert resp.status_code == 400


def test_list_sorted_by_date(client):
    _add(client, days_ago=2, mood=2)
    _add(client, days_ago=0, mood=5)
    _add(client, days_ago=1, mood=3)
    dates = [e["date"] for e in client.get("/entries").get_json()]
    assert dates == sorted(dates)
    assert dates == [_iso(2), _iso(1), _iso(0)]


def test_get_one_entry(client):
    _add(client, days_ago=1, mood=4, note="Fine", tags=["sleep"])
    resp = client.get(f"/entries/{_iso(1)}")
    assert resp.status_code == 200
    assert resp.get_json()["mood"] == 4


def test_get_unknown_date_returns_404(client):
    resp = client.get("/entries/2020-01-01")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_update_entry(client):
    _add(client, days_ago=1, mood=2, note="Bad", tags=["work"])
    resp = client.put(
        f"/entries/{_iso(1)}",
        json={"mood": 5, "note": "Turned around", "tags": ["exercise"]},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["mood"] == 5
    assert body["note"] == "Turned around"
    assert body["tags"] == ["exercise"]
    assert client.get(f"/entries/{_iso(1)}").get_json()["mood"] == 5


def test_update_unknown_returns_404(client):
    resp = client.put("/entries/2020-01-01", json={"mood": 4})
    assert resp.status_code == 404


def test_update_invalid_mood_returns_400(client):
    _add(client, days_ago=1, mood=3)
    resp = client.put(f"/entries/{_iso(1)}", json={"mood": 9})
    assert resp.status_code == 400
    assert client.get(f"/entries/{_iso(1)}").get_json()["mood"] == 3


def test_delete_entry(client):
    _add(client, days_ago=1, mood=3)
    resp = client.delete(f"/entries/{_iso(1)}")
    assert resp.status_code == 200
    assert client.get(f"/entries/{_iso(1)}").status_code == 404
    assert client.get("/entries").get_json() == []


def test_delete_unknown_returns_404(client):
    assert client.delete("/entries/2020-01-01").status_code == 404


def test_weekly_stats_average_math(client):
    _add(client, days_ago=0, mood=5)
    _add(client, days_ago=1, mood=4)
    _add(client, days_ago=2, mood=3)
    _add(client, days_ago=10, mood=1)  # outside the 7-day window
    stats = client.get("/stats/weekly").get_json()
    assert stats["days"] == 7
    assert stats["entries"] == 3
    assert stats["average_mood"] == 4.0
    assert stats["min_mood"] == 3
    assert stats["max_mood"] == 5


def test_weekly_stats_empty_window(client):
    stats = client.get("/stats/weekly").get_json()
    assert stats["entries"] == 0
    assert stats["average_mood"] is None
    assert stats["min_mood"] is None
    assert stats["max_mood"] is None


def test_trend_direction_improving(client):
    _add(client, days_ago=3, mood=2)
    _add(client, days_ago=2, mood=2)
    _add(client, days_ago=1, mood=5)
    _add(client, days_ago=0, mood=5)
    trend = client.get("/stats/trend?days=4").get_json()
    assert trend["days"] == 4
    assert [d["mood"] for d in trend["daily"]] == [2, 2, 5, 5]
    assert trend["first_half_average"] == 2.0
    assert trend["second_half_average"] == 5.0
    assert trend["direction"] == "improving"


def test_trend_direction_declining(client):
    _add(client, days_ago=3, mood=5)
    _add(client, days_ago=2, mood=5)
    _add(client, days_ago=1, mood=1)
    _add(client, days_ago=0, mood=2)
    trend = client.get("/stats/trend?days=4").get_json()
    assert trend["direction"] == "declining"


def test_trend_direction_stable_with_null_halves(client):
    trend = client.get("/stats/trend").get_json()
    assert trend["days"] == 14
    assert trend["daily"] == []
    assert trend["first_half_average"] is None
    assert trend["second_half_average"] is None
    assert trend["direction"] == "stable"


@pytest.mark.parametrize("bad_days", ["0", "-3", "abc", "2.5"])
def test_trend_invalid_days_returns_400(client, bad_days):
    assert client.get(f"/stats/trend?days={bad_days}").status_code == 400


def test_tag_counts_sorted_desc(client):
    _add(client, days_ago=0, mood=4, tags=["work", "exercise"])
    _add(client, days_ago=1, mood=3, tags=["work"])
    _add(client, days_ago=2, mood=5, tags=["sleep", "work"])
    tags = client.get("/stats/tags").get_json()
    assert tags == [
        {"tag": "work", "count": 3},
        {"tag": "exercise", "count": 1},
        {"tag": "sleep", "count": 1},
    ]
