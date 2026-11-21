"""Tests for the kanban_api Flask app, using Flask's test client."""
import pytest

from kanban_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _board(client, name="Website Redesign"):
    return client.post("/boards", json={"name": name})


def _column(client, board_id, name, **extra):
    payload = {"name": name}
    payload.update(extra)
    return client.post(f"/boards/{board_id}/columns", json=payload)


def _card(client, board_id, column_id, title="Do thing", **extra):
    payload = {"title": title}
    payload.update(extra)
    return client.post(
        f"/boards/{board_id}/columns/{column_id}/cards", json=payload
    )


@pytest.fixture
def board(client):
    board = _board(client).get_json()
    todo = _column(client, board["id"], "To Do").get_json()
    doing = _column(client, board["id"], "Doing", wip_limit=2).get_json()
    done = _column(client, board["id"], "Done").get_json()
    return board["id"], todo["id"], doing["id"], done["id"]


def test_list_boards_starts_empty(client):
    assert client.get("/boards").get_json() == []


def test_create_board_returns_201(client):
    resp = _board(client)
    assert resp.status_code == 201
    assert resp.get_json() == {"id": 1, "name": "Website Redesign", "columns": []}


def test_create_board_missing_name_400(client):
    assert client.post("/boards", json={}).status_code == 400
    assert client.post("/boards", json={"name": "  "}).status_code == 400
    assert client.post("/boards", json={"name": 42}).status_code == 400


def test_get_board_404(client):
    assert client.get("/boards/99").status_code == 404


def test_delete_board(client, board):
    board_id = board[0]
    assert client.delete(f"/boards/{board_id}").get_json() == {"deleted": board_id}
    assert client.get(f"/boards/{board_id}").status_code == 404
    assert client.delete("/boards/99").status_code == 404


def test_create_column(client, board):
    board_id = board[0]
    resp = _column(client, board_id, "Review", wip_limit=3)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["name"] == "Review"
    assert body["wip_limit"] == 3
    assert body["cards"] == []


def test_create_column_defaults_unlimited(client, board):
    body = _column(client, board[0], "Backlog").get_json()
    assert body["wip_limit"] is None


def test_create_column_duplicate_name_409(client, board):
    assert _column(client, board[0], "to do").status_code == 409
    assert _column(client, board[0], "To Do").status_code == 409


def test_create_column_bad_input(client, board):
    assert _column(client, board[0], "").status_code == 400
    assert _column(client, board[0], "X", wip_limit=-1).status_code == 400
    assert _column(client, board[0], "X", wip_limit=1.5).status_code == 400
    assert client.post("/boards/99/columns", json={"name": "X"}).status_code == 404


def test_update_column_rename_and_limit(client, board):
    board_id, _todo, doing_id, _done = board
    body = client.put(
        f"/boards/{board_id}/columns/{doing_id}",
        json={"name": "In Progress", "wip_limit": 5},
    ).get_json()
    assert body["name"] == "In Progress"
    assert body["wip_limit"] == 5


def test_update_column_duplicate_name_409(client, board):
    board_id, _todo, doing_id, _done = board
    resp = client.put(
        f"/boards/{board_id}/columns/{doing_id}", json={"name": "To Do"}
    )
    assert resp.status_code == 409


def test_update_column_limit_below_card_count_409(client, board):
    board_id, todo_id, _doing, _done = board
    _card(client, board_id, todo_id, "A")
    _card(client, board_id, todo_id, "B")
    resp = client.put(
        f"/boards/{board_id}/columns/{todo_id}", json={"wip_limit": 1}
    )
    assert resp.status_code == 409
    assert client.put(
        f"/boards/{board_id}/columns/{todo_id}", json={"wip_limit": 2}
    ).status_code == 200


def test_update_column_nothing_400(client, board):
    board_id, todo_id, _doing, _done = board
    assert client.put(
        f"/boards/{board_id}/columns/{todo_id}", json={}
    ).status_code == 400
    assert client.put(
        "/boards/99/columns/1", json={"name": "X"}
    ).status_code == 404


def test_delete_empty_column(client, board):
    board_id, todo_id, _doing, _done = board
    assert client.delete(
        f"/boards/{board_id}/columns/{todo_id}"
    ).get_json() == {"deleted": todo_id}
    assert client.delete(
        f"/boards/{board_id}/columns/{todo_id}"
    ).status_code == 404


def test_delete_nonempty_column_409(client, board):
    board_id, todo_id, _doing, _done = board
    _card(client, board_id, todo_id, "A")
    resp = client.delete(f"/boards/{board_id}/columns/{todo_id}")
    assert resp.status_code == 409


def test_create_card_defaults(client, board):
    board_id, todo_id, _doing, _done = board
    resp = _card(client, board_id, todo_id, "Design homepage")
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["title"] == "Design homepage"
    assert body["description"] == ""
    assert body["assignee"] is None
    assert body["priority"] == "medium"
    assert body["column_id"] == todo_id
    assert body["position"] == 0


def test_create_card_with_fields(client, board):
    board_id, todo_id, _doing, _done = board
    body = _card(
        client, board_id, todo_id, "Write copy",
        description="Landing page", assignee="Usman", priority="high",
    ).get_json()
    assert body["description"] == "Landing page"
    assert body["assignee"] == "Usman"
    assert body["priority"] == "high"


def test_create_card_positions_increment(client, board):
    board_id, todo_id, _doing, _done = board
    first = _card(client, board_id, todo_id, "A").get_json()
    second = _card(client, board_id, todo_id, "B").get_json()
    assert (first["position"], second["position"]) == (0, 1)


def test_create_card_bad_input(client, board):
    board_id, todo_id, _doing, _done = board
    assert _card(client, board_id, todo_id, "").status_code == 400
    assert _card(client, board_id, todo_id, "X", priority="urgent").status_code == 400
    assert _card(client, board_id, todo_id, "X", assignee=" ").status_code == 400
    assert _card(client, board_id, 99, "X").status_code == 404


def test_create_card_wip_limit_409(client, board):
    board_id, _todo, doing_id, _done = board
    _card(client, board_id, doing_id, "A")
    _card(client, board_id, doing_id, "B")
    resp = _card(client, board_id, doing_id, "C")
    assert resp.status_code == 409
    assert "WIP limit" in resp.get_json()["error"]


def test_get_and_update_card(client, board):
    board_id, todo_id, _doing, _done = board
    card = _card(client, board_id, todo_id, "A").get_json()
    fetched = client.get(f"/boards/{board_id}/cards/{card['id']}").get_json()
    assert fetched["title"] == "A"
    updated = client.put(
        f"/boards/{board_id}/cards/{card['id']}",
        json={"title": "A2", "priority": "high", "assignee": "Usman"},
    ).get_json()
    assert updated["title"] == "A2"
    assert updated["priority"] == "high"
    assert updated["assignee"] == "Usman"
    assert client.get("/boards/99/cards/1").status_code == 404
    assert client.get(f"/boards/{board_id}/cards/99").status_code == 404


def test_update_card_bad_input(client, board):
    board_id, todo_id, _doing, _done = board
    card = _card(client, board_id, todo_id, "A").get_json()
    assert client.put(
        f"/boards/{board_id}/cards/{card['id']}", json={}
    ).status_code == 400
    assert client.put(
        f"/boards/{board_id}/cards/{card['id']}",
        json={"priority": "urgent"},
    ).status_code == 400


def test_delete_card_renumbers(client, board):
    board_id, todo_id, _doing, _done = board
    first = _card(client, board_id, todo_id, "A").get_json()
    second = _card(client, board_id, todo_id, "B").get_json()
    assert client.delete(
        f"/boards/{board_id}/cards/{first['id']}"
    ).get_json() == {"deleted": first["id"]}
    remaining = client.get(f"/boards/{board_id}/cards/{second['id']}").get_json()
    assert remaining["position"] == 0


def test_move_card_between_columns(client, board):
    board_id, todo_id, doing_id, done_id = board
    card = _card(client, board_id, todo_id, "A").get_json()
    moved = client.post(
        f"/boards/{board_id}/cards/{card['id']}/move",
        json={"column_id": doing_id},
    ).get_json()
    assert moved["column_id"] == doing_id
    assert moved["position"] == 0
    moved = client.post(
        f"/boards/{board_id}/cards/{card['id']}/move",
        json={"column_id": done_id, "position": 0},
    ).get_json()
    assert moved["column_id"] == done_id
    assert moved["position"] == 0


def test_move_card_with_position(client, board):
    board_id, todo_id, doing_id, _done = board
    first = _card(client, board_id, todo_id, "A").get_json()
    second = _card(client, board_id, todo_id, "B").get_json()
    third = _card(client, board_id, todo_id, "C").get_json()
    client.post(
        f"/boards/{board_id}/cards/{first['id']}/move",
        json={"column_id": doing_id},
    )
    client.post(
        f"/boards/{board_id}/cards/{second['id']}/move",
        json={"column_id": doing_id},
    )
    # Move C into Doing at position 0; WIP is 2 so this must fail.
    assert client.post(
        f"/boards/{board_id}/cards/{third['id']}/move",
        json={"column_id": doing_id, "position": 0},
    ).status_code == 409
    # Move A back to To Do at position 0.
    moved = client.post(
        f"/boards/{board_id}/cards/{first['id']}/move",
        json={"column_id": todo_id, "position": 0},
    ).get_json()
    assert moved["position"] == 0
    staying = client.get(f"/boards/{board_id}/cards/{third['id']}").get_json()
    assert staying["position"] == 1  # C shifted after A left, then A inserted


def test_move_card_wip_limit_409(client, board):
    board_id, todo_id, doing_id, _done = board
    _card(client, board_id, doing_id, "A")
    _card(client, board_id, doing_id, "B")
    card = _card(client, board_id, todo_id, "C").get_json()
    resp = client.post(
        f"/boards/{board_id}/cards/{card['id']}/move",
        json={"column_id": doing_id},
    )
    assert resp.status_code == 409


def test_move_card_bad_input(client, board):
    board_id, todo_id, doing_id, _done = board
    card = _card(client, board_id, todo_id, "A").get_json()
    assert client.post(
        f"/boards/{board_id}/cards/{card['id']}/move", json={}
    ).status_code == 400
    assert client.post(
        f"/boards/{board_id}/cards/{card['id']}/move",
        json={"column_id": 99},
    ).status_code == 404
    assert client.post(
        f"/boards/{board_id}/cards/{card['id']}/move",
        json={"column_id": doing_id, "position": -1},
    ).status_code == 400
    assert client.post(
        f"/boards/{board_id}/cards/99/move",
        json={"column_id": doing_id},
    ).status_code == 404


def test_board_stats(client, board):
    board_id, todo_id, doing_id, done_id = board
    _card(client, board_id, todo_id, "A", priority="high")
    _card(client, board_id, todo_id, "B", priority="low")
    card = _card(client, board_id, todo_id, "C", priority="medium").get_json()
    client.post(
        f"/boards/{board_id}/cards/{card['id']}/move",
        json={"column_id": done_id},
    )
    stats = client.get(f"/boards/{board_id}/stats").get_json()
    assert stats["board_id"] == board_id
    assert stats["total_cards"] == 3
    assert stats["by_priority"] == {"low": 1, "medium": 1, "high": 1}
    by_name = {c["name"]: c for c in stats["columns"]}
    assert by_name["To Do"]["cards"] == 2
    assert by_name["Doing"]["cards"] == 0
    assert by_name["Doing"]["wip_limit"] == 2
    assert by_name["Doing"]["wip_used_pct"] == 0.0
    assert by_name["Done"]["cards"] == 1
    assert by_name["Done"]["wip_used_pct"] is None
    assert client.get("/boards/99/stats").status_code == 404
