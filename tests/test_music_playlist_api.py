"""Tests for the music_playlist_api Flask app, using Flask's test client."""
import pytest

from music_playlist_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _playlist(client, name="Road Trip", **extra):
    payload = {"name": name}
    payload.update(extra)
    return client.post("/playlists", json=payload)


def _track(client, pid, title="Song", artist="Artist", duration=180, **extra):
    payload = {"title": title, "artist": artist,
               "duration_seconds": duration}
    payload.update(extra)
    return client.post(f"/playlists/{pid}/tracks", json=payload)


def _setup(client, public=True):
    pid = _playlist(client, public=public).get_json()["id"]
    a = _track(client, pid, "Blinding Lights", "The Weeknd", 200,
               album="After Hours").get_json()
    b = _track(client, pid, "Levitating", "Dua Lipa", 203,
               album="Future Nostalgia").get_json()
    c = _track(client, pid, "As It Was", "Harry Styles", 167).get_json()
    return pid, a, b, c


def test_list_playlists_starts_empty(client):
    assert client.get("/playlists").get_json() == []


def test_create_playlist_returns_201(client):
    resp = _playlist(client, "Chill Vibes", description="Lo-fi",
                     public=True)
    assert resp.status_code == 201
    playlist = resp.get_json()
    assert playlist == {
        "id": 1, "name": "Chill Vibes", "description": "Lo-fi",
        "public": True, "tracks": [], "track_count": 0,
        "total_duration_seconds": 0}


def test_create_playlist_defaults(client):
    playlist = _playlist(client, "Private Mix").get_json()
    assert playlist["public"] is False
    assert playlist["description"] == ""


def test_create_playlist_validation_errors(client):
    assert _playlist(client, "").status_code == 400
    assert _playlist(client, "   ").status_code == 400
    assert _playlist(client, 42).status_code == 400
    assert _playlist(client, "X", description=5).status_code == 400
    assert _playlist(client, "X", public="yes").status_code == 400
    assert client.post("/playlists", json=[1]).status_code == 400


def test_get_playlist_and_404(client):
    pid = _playlist(client, "Found").get_json()["id"]
    assert client.get(f"/playlists/{pid}").get_json()["name"] == "Found"
    assert client.get("/playlists/999").status_code == 404


def test_list_playlists_public_filter(client):
    _playlist(client, "Open", public=True)
    _playlist(client, "Closed", public=False)
    public = client.get("/playlists",
                        query_string={"public": "true"}).get_json()
    assert [p["name"] for p in public] == ["Open"]
    private = client.get("/playlists",
                         query_string={"public": "false"}).get_json()
    assert [p["name"] for p in private] == ["Closed"]
    assert client.get("/playlists",
                      query_string={"public": "maybe"}).status_code == 400


def test_update_playlist(client):
    pid = _playlist(client, "Old").get_json()["id"]
    updated = client.put(f"/playlists/{pid}",
                         json={"name": "New", "public": True}).get_json()
    assert updated["name"] == "New"
    assert updated["public"] is True
    assert client.put(f"/playlists/{pid}", json={"name": ""}).status_code == 400
    assert client.put(f"/playlists/{pid}",
                      json={"public": 1}).status_code == 400
    assert client.put("/playlists/999", json={"name": "X"}).status_code == 404


def test_add_track_positions(client):
    pid, a, b, c = _setup(client)
    assert (a["position"], b["position"], c["position"]) == (0, 1, 2)
    assert a["album"] == "After Hours"
    assert c["album"] == ""


def test_add_track_validation_errors(client):
    pid = _playlist(client, "V").get_json()["id"]
    assert _track(client, pid, "", "A", 100).status_code == 400
    assert _track(client, pid, "T", "", 100).status_code == 400
    assert _track(client, pid, "T", "A", 0).status_code == 400
    assert _track(client, pid, "T", "A", -5).status_code == 400
    assert _track(client, pid, "T", "A", 90.5).status_code == 400
    assert _track(client, pid, "T", "A", True).status_code == 400
    assert _track(client, pid, "T", "A", 100, album=9).status_code == 400
    assert _track(client, 999, "T", "A", 100).status_code == 404


def test_duplicate_track_conflict(client):
    pid, _, _, _ = _setup(client)
    assert _track(client, pid, "blinding lights", "the weeknd",
                  200).status_code == 409
    assert _track(client, pid, "Blinding Lights", "Someone Else",
                  200).status_code == 201


def test_list_and_get_tracks(client):
    pid, a, b, c = _setup(client)
    titles = [t["title"] for t in
              client.get(f"/playlists/{pid}/tracks").get_json()]
    assert titles == ["Blinding Lights", "Levitating", "As It Was"]
    got = client.get(f"/playlists/{pid}/tracks/{b['id']}").get_json()
    assert got["artist"] == "Dua Lipa"
    assert got["duration_seconds"] == 203
    assert client.get(f"/playlists/{pid}/tracks/999").status_code == 404
    assert client.get("/playlists/999/tracks").status_code == 404


def test_playlist_view_totals(client):
    pid, _, _, _ = _setup(client)
    view = client.get(f"/playlists/{pid}").get_json()
    assert view["track_count"] == 3
    assert view["total_duration_seconds"] == 570
    assert [t["position"] for t in view["tracks"]] == [0, 1, 2]


def test_summary_format(client):
    pid, _, _, _ = _setup(client)
    summary = client.get(f"/playlists/{pid}/summary").get_json()
    assert summary == {
        "playlist_id": pid, "track_count": 3,
        "total_duration_seconds": 570, "total_duration": "9:30"}
    empty = _playlist(client, "Empty").get_json()["id"]
    assert client.get(f"/playlists/{empty}/summary").get_json()[
        "total_duration"] == "0:00"
    long_pid = _playlist(client, "Long").get_json()["id"]
    _track(client, long_pid, "Epic", "Band", 3723)
    assert client.get(f"/playlists/{long_pid}/summary").get_json()[
        "total_duration"] == "1:02:03"
    assert client.get("/playlists/999/summary").status_code == 404


def test_reorder_track(client):
    pid, a, b, c = _setup(client)
    moved = client.put(f"/playlists/{pid}/tracks/{c['id']}",
                       json={"position": 0}).get_json()
    assert moved["position"] == 0
    titles = [t["title"] for t in
              client.get(f"/playlists/{pid}/tracks").get_json()]
    assert titles == ["As It Was", "Blinding Lights", "Levitating"]
    positions = [t["position"] for t in
                 client.get(f"/playlists/{pid}/tracks").get_json()]
    assert positions == [0, 1, 2]


def test_reorder_validation_errors(client):
    pid, a, _, _ = _setup(client)
    assert client.put(f"/playlists/{pid}/tracks/{a['id']}",
                      json={"position": 3}).status_code == 400
    assert client.put(f"/playlists/{pid}/tracks/{a['id']}",
                      json={"position": -1}).status_code == 400
    assert client.put(f"/playlists/{pid}/tracks/{a['id']}",
                      json={"position": "first"}).status_code == 400
    assert client.put(f"/playlists/{pid}/tracks/999",
                      json={"position": 0}).status_code == 404


def test_update_track_fields(client):
    pid, a, _, _ = _setup(client)
    updated = client.put(f"/playlists/{pid}/tracks/{a['id']}",
                         json={"duration_seconds": 210,
                               "album": "After Hours (Deluxe)"}).get_json()
    assert updated["duration_seconds"] == 210
    assert updated["album"] == "After Hours (Deluxe)"
    assert client.put(f"/playlists/{pid}/tracks/{a['id']}",
                      json={"duration_seconds": 0}).status_code == 400
    assert client.put(f"/playlists/{pid}/tracks/{a['id']}",
                      json={"title": ""}).status_code == 400


def test_update_track_to_duplicate_conflict(client):
    pid, a, b, _ = _setup(client)
    assert client.put(f"/playlists/{pid}/tracks/{b['id']}",
                      json={"title": "Blinding Lights",
                            "artist": "The Weeknd"}).status_code == 409


def test_delete_track_reindexes(client):
    pid, a, b, c = _setup(client)
    resp = client.delete(f"/playlists/{pid}/tracks/{b['id']}")
    assert resp.status_code == 200
    assert resp.get_json() == {"deleted": b["id"]}
    tracks = client.get(f"/playlists/{pid}/tracks").get_json()
    assert [t["title"] for t in tracks] == ["Blinding Lights", "As It Was"]
    assert [t["position"] for t in tracks] == [0, 1]
    summary = client.get(f"/playlists/{pid}/summary").get_json()
    assert summary["track_count"] == 2
    assert summary["total_duration_seconds"] == 367
    assert client.delete(f"/playlists/{pid}/tracks/{b['id']}").status_code == 404


def test_delete_playlist_removes_tracks(client):
    pid, a, _, _ = _setup(client)
    assert client.delete(f"/playlists/{pid}").status_code == 200
    assert client.get(f"/playlists/{pid}").status_code == 404
    assert client.get(f"/playlists/{pid}/tracks").status_code == 404
    assert client.delete("/playlists/999").status_code == 404
