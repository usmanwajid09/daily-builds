"""Tests for the chat_api Flask app, using Flask's test client."""
import pytest

from chat_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _room(client, name="general"):
    return client.post("/rooms", json={"name": name})


def _msg(client, room_id, sender="usman", text="hi"):
    return client.post(
        f"/rooms/{room_id}/messages", json={"sender": sender, "text": text}
    )


def test_list_rooms_starts_empty(client):
    assert client.get("/rooms").get_json() == []


def test_create_room_returns_201(client):
    resp = _room(client, "general")
    assert resp.status_code == 201
    assert resp.get_json() == {"id": 1, "name": "general", "message_count": 0}


def test_create_room_requires_non_empty_name(client):
    assert _room(client, "").status_code == 400
    assert _room(client, "   ").status_code == 400
    assert client.post("/rooms", json={}).status_code == 400
    assert client.post("/rooms", json={"name": 42}).status_code == 400


def test_get_room_and_404(client):
    room_id = _room(client).get_json()["id"]
    assert client.get(f"/rooms/{room_id}").get_json()["name"] == "general"
    assert client.get("/rooms/999").status_code == 404


def test_post_message_returns_201(client):
    room_id = _room(client).get_json()["id"]
    resp = _msg(client, room_id, "usman", "Hello!")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1, "room_id": room_id, "sender": "usman", "text": "Hello!",
    }


def test_post_message_validation(client):
    room_id = _room(client).get_json()["id"]
    assert _msg(client, room_id, "", "hi").status_code == 400
    assert _msg(client, room_id, "usman", "   ").status_code == 400
    assert _msg(client, room_id, "usman", 123).status_code == 400
    assert client.post(f"/rooms/{room_id}/messages", json={}).status_code == 400


def test_post_message_to_missing_room_is_404(client):
    assert _msg(client, 999).status_code == 404
    assert client.get("/rooms/999/messages").status_code == 404


def test_history_is_per_room_and_ordered(client):
    a = _room(client, "a").get_json()["id"]
    b = _room(client, "b").get_json()["id"]
    _msg(client, a, "usman", "first")
    _msg(client, b, "ali", "other room")
    _msg(client, a, "ali", "second")
    history = client.get(f"/rooms/{a}/messages").get_json()
    assert [m["text"] for m in history] == ["first", "second"]
    assert client.get(f"/rooms/{b}/messages").get_json()[0]["text"] == "other room"


def test_history_sender_filter_and_limit(client):
    room_id = _room(client).get_json()["id"]
    _msg(client, room_id, "usman", "one")
    _msg(client, room_id, "ali", "two")
    _msg(client, room_id, "usman", "three")
    mine = client.get(
        f"/rooms/{room_id}/messages", query_string={"sender": "usman"}
    ).get_json()
    assert [m["text"] for m in mine] == ["one", "three"]
    first_two = client.get(
        f"/rooms/{room_id}/messages", query_string={"limit": 2}
    ).get_json()
    assert [m["text"] for m in first_two] == ["one", "two"]
    assert client.get(
        f"/rooms/{room_id}/messages", query_string={"limit": "abc"}
    ).status_code == 400
    assert client.get(
        f"/rooms/{room_id}/messages", query_string={"limit": -1}
    ).status_code == 400


def test_room_message_count_updates(client):
    room_id = _room(client).get_json()["id"]
    _msg(client, room_id)
    _msg(client, room_id, "ali", "two")
    assert client.get(f"/rooms/{room_id}").get_json()["message_count"] == 2


def test_delete_message(client):
    room_id = _room(client).get_json()["id"]
    mid = _msg(client, room_id).get_json()["id"]
    assert client.delete(f"/rooms/{room_id}/messages/{mid}").get_json() == {
        "deleted": mid
    }
    assert client.get(f"/rooms/{room_id}/messages").get_json() == []
    assert client.delete(f"/rooms/{room_id}/messages/{mid}").status_code == 404
    assert client.delete("/rooms/999/messages/1").status_code == 404


def test_delete_room_removes_its_messages(client):
    room_id = _room(client).get_json()["id"]
    _msg(client, room_id)
    resp = client.delete(f"/rooms/{room_id}")
    assert resp.get_json() == {"deleted": room_id, "messages_deleted": 1}
    assert client.get(f"/rooms/{room_id}").status_code == 404
    assert client.delete(f"/rooms/{room_id}").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
