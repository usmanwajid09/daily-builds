"""Tests for the journal_api Flask app, using Flask's test client."""
import pytest

from journal_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _entry(client, day="2026-11-20", mood=4, text="A good day.", tags=None):
    body = {"date": day, "mood": mood, "text": text}
    if tags is not None:
        body["tags"] = tags
    return client.post("/entries", json=body)


def _seed(client):
    _entry(client, "2026-11-20", 4, "Shipped Docker.", ["work", "docker"])
    _entry(client, "2026-11-22", 3, "Reviewed PRs.", ["work"])
    _entry(client, "2026-11-24", 5, "Morning run.", ["health", "study"])
    _entry(client, "2026-11-26", 4, "Finished APIs.", ["work", "study"])


def test_list_starts_empty(client):
    assert client.get("/entries").get_json() == []


def test_create_entry_returns_201(client):
    resp = _entry(client, "2026-11-20", 4, "A good day.", ["work"])
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "date": "2026-11-20",
        "mood": 4,
        "text": "A good day.",
        "tags": ["work"],
    }


def test_create_entry_defaults_to_no_tags(client):
    resp = _entry(client)
    assert resp.status_code == 201
    assert resp.get_json()["tags"] == []


def test_mood_out_of_range_returns_400(client):
    assert _entry(client, mood=0).status_code == 400
    assert _entry(client, mood=6).status_code == 400
    assert _entry(client, mood=7).status_code == 400
    assert _entry(client, mood="happy").status_code == 400
    assert _entry(client, mood=3.5).status_code == 400
    assert _entry(client, mood=True).status_code == 400


def test_bad_date_returns_400(client):
    assert _entry(client, day="not-a-date").status_code == 400
    assert _entry(client, day="2026-02-30").status_code == 400
    assert _entry(client, day="20-11-2026").status_code == 400


def test_missing_text_returns_400(client):
    assert _entry(client, text="").status_code == 400
    assert _entry(client, text="   ").status_code == 400
    assert client.post("/entries", json={"date": "2026-11-20", "mood": 4}).status_code == 400


def test_bad_tags_returns_400(client):
    assert _entry(client, tags="work").status_code == 400
    assert _entry(client, tags=["work", 5]).status_code == 400
    assert _entry(client, tags=[""]).status_code == 400


def test_get_entry_and_404(client):
    entry_id = _entry(client).get_json()["id"]
    assert client.get(f"/entries/{entry_id}").get_json()["mood"] == 4
    assert client.get("/entries/999").status_code == 404


def test_list_entries_in_date_range(client):
    _seed(client)
    week = client.get(
        "/entries", query_string={"from": "2026-11-23", "to": "2026-11-26"}
    ).get_json()
    assert [e["date"] for e in week] == ["2026-11-24", "2026-11-26"]
    assert len(client.get("/entries").get_json()) == 4


def test_list_entries_bad_range_returns_400(client):
    _seed(client)
    assert client.get("/entries", query_string={"from": "bad"}).status_code == 400
    assert client.get(
        "/entries", query_string={"from": "2026-11-26", "to": "2026-11-20"}
    ).status_code == 400


def test_mood_average_over_range(client):
    _seed(client)
    stats = client.get(
        "/stats/mood-average",
        query_string={"from": "2026-11-20", "to": "2026-11-26"},
    ).get_json()
    assert stats == {"count": 4, "average_mood": 4.0}
    partial = client.get(
        "/stats/mood-average",
        query_string={"from": "2026-11-20", "to": "2026-11-22"},
    ).get_json()
    assert partial == {"count": 2, "average_mood": 3.5}


def test_mood_average_empty_range(client):
    _seed(client)
    stats = client.get(
        "/stats/mood-average",
        query_string={"from": "2026-01-01", "to": "2026-01-31"},
    ).get_json()
    assert stats == {"count": 0, "average_mood": None}


def test_search_by_tag(client):
    _seed(client)
    work = client.get("/entries/search", query_string={"tag": "work"}).get_json()
    assert [e["date"] for e in work] == ["2026-11-20", "2026-11-22", "2026-11-26"]
    # Tag search is case-insensitive.
    study = client.get("/entries/search", query_string={"tag": "STUDY"}).get_json()
    assert len(study) == 2
    assert client.get("/entries/search", query_string={"tag": "nope"}).get_json() == []


def test_search_without_tag_returns_400(client):
    assert client.get("/entries/search").status_code == 400


def test_delete_entry(client):
    entry_id = _entry(client).get_json()["id"]
    assert client.delete(f"/entries/{entry_id}").get_json() == {"deleted": entry_id}
    assert client.get(f"/entries/{entry_id}").status_code == 404
    assert client.delete("/entries/999").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
