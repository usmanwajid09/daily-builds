"""Tests for the sleep_tracker_api Flask app, using Flask's test client."""
import pytest

from sleep_tracker_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _night(client, date="2026-11-21", bedtime="23:30", wake_time="07:15",
           quality=4, **extra):
    payload = {"date": date, "bedtime": bedtime, "wake_time": wake_time,
               "quality": quality}
    payload.update(extra)
    return client.post("/nights", json=payload)


def test_list_starts_empty(client):
    assert client.get("/nights").get_json() == []


def test_create_night_returns_201_and_duration(client):
    resp = _night(client, notes="Slept well")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "date": "2026-11-21",
        "bedtime": "23:30",
        "wake_time": "07:15",
        "duration_minutes": 465,
        "duration_hours": 7.75,
        "quality": 4,
        "notes": "Slept well",
    }


def test_duration_same_day(client):
    assert _night(client, bedtime="22:00", wake_time="23:30").get_json()[
        "duration_minutes"] == 90


def test_duration_crosses_midnight(client):
    assert _night(client, bedtime="00:15", wake_time="07:00").get_json()[
        "duration_minutes"] == 405


def test_notes_default_none(client):
    assert _night(client).get_json()["notes"] is None


def test_create_missing_date_400(client):
    resp = client.post("/nights", json={"bedtime": "23:30",
                                        "wake_time": "07:15", "quality": 4})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_create_bad_date_400(client):
    assert _night(client, date="21-11-2026").status_code == 400
    assert _night(client, date="2026-02-30").status_code == 400


def test_create_bad_time_400(client):
    assert _night(client, bedtime="25:00").status_code == 400
    assert _night(client, wake_time="07:15:00").status_code == 400
    assert _night(client, bedtime="23:60").status_code == 400


def test_create_zero_duration_400(client):
    assert _night(client, bedtime="07:00", wake_time="07:00").status_code == 400


def test_create_quality_out_of_range_400(client):
    assert _night(client, quality=0).status_code == 400
    assert _night(client, quality=6).status_code == 400
    assert _night(client, quality="good").status_code == 400


def test_create_non_json_body_400(client):
    resp = client.post("/nights", data="not json",
                       content_type="text/plain")
    assert resp.status_code == 400


def test_duplicate_date_409(client):
    assert _night(client).status_code == 201
    resp = _night(client, bedtime="22:00", wake_time="06:00", quality=3)
    assert resp.status_code == 409
    assert "error" in resp.get_json()


def test_get_night(client):
    _night(client)
    resp = client.get("/nights/1")
    assert resp.status_code == 200
    assert resp.get_json()["date"] == "2026-11-21"


def test_get_missing_night_404(client):
    assert client.get("/nights/99").status_code == 404


def test_list_sorted_newest_first(client):
    _night(client, date="2026-11-19")
    _night(client, date="2026-11-21")
    _night(client, date="2026-11-20")
    dates = [n["date"] for n in client.get("/nights").get_json()]
    assert dates == ["2026-11-21", "2026-11-20", "2026-11-19"]


def test_filter_by_min_quality(client):
    _night(client, date="2026-11-19", quality=2)
    _night(client, date="2026-11-20", quality=5)
    _night(client, date="2026-11-21", quality=4)
    rows = client.get("/nights?min_quality=4").get_json()
    assert [n["date"] for n in rows] == ["2026-11-21", "2026-11-20"]


def test_filter_bad_min_quality_400(client):
    assert client.get("/nights?min_quality=9").status_code == 400
    assert client.get("/nights?min_quality=abc").status_code == 400


def test_filter_by_date(client):
    _night(client, date="2026-11-19")
    _night(client, date="2026-11-20")
    rows = client.get("/nights?date=2026-11-19").get_json()
    assert len(rows) == 1 and rows[0]["date"] == "2026-11-19"


def test_update_night_recomputes_duration(client):
    _night(client)
    resp = client.put("/nights/1", json={"wake_time": "08:15"})
    assert resp.status_code == 200
    assert resp.get_json()["duration_minutes"] == 525


def test_update_to_duplicate_date_409(client):
    _night(client, date="2026-11-19")
    _night(client, date="2026-11-20")
    assert client.put("/nights/2", json={"date": "2026-11-19"}).status_code == 409


def test_update_nothing_400(client):
    _night(client)
    assert client.put("/nights/1", json={}).status_code == 400


def test_update_missing_404(client):
    assert client.put("/nights/99", json={"quality": 5}).status_code == 404


def test_delete_night(client):
    _night(client)
    assert client.delete("/nights/1").get_json() == {"deleted": 1}
    assert client.get("/nights/1").status_code == 404


def test_delete_missing_404(client):
    assert client.delete("/nights/99").status_code == 404


def test_stats_empty(client):
    stats = client.get("/stats").get_json()
    assert stats["total_nights"] == 0
    assert stats["average_duration_minutes"] is None
    assert stats["streak_days"] == 0


def test_stats_averages_and_streak(client):
    _night(client, date="2026-11-19", bedtime="23:00", wake_time="07:00",
           quality=5)  # 480 min
    _night(client, date="2026-11-20", bedtime="23:30", wake_time="07:30",
           quality=3)  # 480 min
    _night(client, date="2026-11-21", bedtime="23:00", wake_time="06:00",
           quality=4)  # 420 min
    stats = client.get("/stats").get_json()
    assert stats["total_nights"] == 3
    assert stats["total_sleep_minutes"] == 1380
    assert stats["average_duration_minutes"] == 460.0
    assert stats["average_duration_hours"] == 7.67
    assert stats["average_quality"] == 4.0
    assert stats["longest_night_minutes"] == 480
    assert stats["shortest_night_minutes"] == 420
    assert stats["streak_days"] == 3


def test_stats_streak_breaks_on_gap(client):
    _night(client, date="2026-11-18")
    _night(client, date="2026-11-20")
    _night(client, date="2026-11-21")
    assert client.get("/stats").get_json()["streak_days"] == 2


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404
