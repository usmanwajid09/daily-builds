"""Flask kanban board API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /boards`` — create a board (``{\"name\": str}``)
- ``GET /boards`` — list boards
- ``GET /boards/<id>`` — fetch one board (with its columns and cards)
- ``DELETE /boards/<id>`` — delete a board and everything in it
- ``POST /boards/<id>/columns`` — add a column (``{\"name\": str,
  \"wip_limit\": int | null?}``; null means unlimited)
- ``PUT /boards/<id>/columns/<col_id>`` — rename a column and/or
  change its WIP limit (409 if the new limit is below the current
  card count)
- ``DELETE /boards/<id>/columns/<col_id>`` — delete an empty column
  (409 if it still holds cards)
- ``POST /boards/<id>/columns/<col_id>/cards`` — add a card
  (``{\"title\": str, \"description\": str?, \"assignee\": str?,
  \"priority\": \"low\" | \"medium\" | \"high\"?}``; 409 if the column
  is at its WIP limit)
- ``GET /boards/<id>/cards/<card_id>`` — fetch one card
- ``PUT /boards/<id>/cards/<card_id>`` — update a card's title,
  description, assignee and/or priority
- ``DELETE /boards/<id>/cards/<card_id>`` — delete a card
- ``POST /boards/<id>/cards/<card_id>/move`` — move a card
  (``{\"column_id\": int, \"position\": int?}``; 409 if the target
  column is at its WIP limit)
- ``GET /boards/<id>/stats`` — card counts per column, WIP usage,
  total cards

A board is ``{\"id\": int, \"name\": str, \"columns\": [...]}``; a
column is ``{\"id\": int, \"name\": str, \"wip_limit\": int | None,
\"cards\": [...]}``; a card is ``{\"id\": int, \"title\": str,
\"description\": str, \"assignee\": str | None,
\"priority\": \"low\" | \"medium\" | \"high\", \"column_id\": int,
\"position\": int}``. Cards are ordered by ``position`` within a
column (0-based, contiguous).
"""
from flask import Flask, jsonify, request

PRIORITIES = ("low", "medium", "high")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def create_app():
    """Return a Flask app with a fresh, empty in-memory kanban store."""
    app = Flask(__name__)
    app.config["BOARDS"] = {}
    app.config["NEXT_BOARD_ID"] = 1
    app.config["NEXT_COLUMN_ID"] = 1
    app.config["NEXT_CARD_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_board(board_id):
        board = app.config["BOARDS"].get(board_id)
        if board is None:
            return None, _error(f"board {board_id} not found", 404)
        return board, None

    def _get_column(board, column_id):
        for column in board["columns"]:
            if column["id"] == column_id:
                return column, None
        return None, _error(
            f"column {column_id} not found in board {board['id']}", 404
        )

    def _find_card(board, card_id):
        for column in board["columns"]:
            for card in column["cards"]:
                if card["id"] == card_id:
                    return card, column
        return None, None

    def _public_card(card):
        return dict(card)

    def _public_column(column):
        return {
            "id": column["id"],
            "name": column["name"],
            "wip_limit": column["wip_limit"],
            "cards": [_public_card(c) for c in column["cards"]],
        }

    def _public_board(board):
        return {
            "id": board["id"],
            "name": board["name"],
            "columns": [_public_column(c) for c in board["columns"]],
        }

    def _renumber(column):
        for index, card in enumerate(column["cards"]):
            card["position"] = index

    @app.post("/boards")
    def create_board():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        board_id = app.config["NEXT_BOARD_ID"]
        app.config["NEXT_BOARD_ID"] += 1
        board = {"id": board_id, "name": name.strip(), "columns": []}
        app.config["BOARDS"][board_id] = board
        return jsonify(_public_board(board)), 201

    @app.get("/boards")
    def list_boards():
        boards = sorted(app.config["BOARDS"].values(), key=lambda b: b["id"])
        return jsonify([_public_board(b) for b in boards])

    @app.get("/boards/<int:board_id>")
    def get_board(board_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        return jsonify(_public_board(board))

    @app.delete("/boards/<int:board_id>")
    def delete_board(board_id):
        _board, err = _get_board(board_id)
        if err is not None:
            return err
        del app.config["BOARDS"][board_id]
        return jsonify({"deleted": board_id})

    @app.post("/boards/<int:board_id>/columns")
    def create_column(board_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        wip_limit = data.get("wip_limit")
        if wip_limit is not None and (
            not _is_int(wip_limit) or wip_limit < 0
        ):
            return _error("wip_limit must be a non-negative integer or null", 400)
        if any(c["name"].lower() == name.strip().lower()
               for c in board["columns"]):
            return _error(
                f"board {board_id} already has a column named "
                f"{name.strip()!r}",
                409,
            )
        column_id = app.config["NEXT_COLUMN_ID"]
        app.config["NEXT_COLUMN_ID"] += 1
        column = {
            "id": column_id,
            "name": name.strip(),
            "wip_limit": wip_limit,
            "cards": [],
        }
        board["columns"].append(column)
        return jsonify(_public_column(column)), 201

    @app.put("/boards/<int:board_id>/columns/<int:column_id>")
    def update_column(board_id, column_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        column, err = _get_column(board, column_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "name" not in data and "wip_limit" not in data:
            return _error("nothing to update: provide name and/or wip_limit", 400)
        if "name" in data:
            name = data["name"]
            if not isinstance(name, str) or not name.strip():
                return _error("name must be a non-empty string", 400)
            if any(c["id"] != column_id
                   and c["name"].lower() == name.strip().lower()
                   for c in board["columns"]):
                return _error(
                    f"board {board_id} already has a column named "
                    f"{name.strip()!r}",
                    409,
                )
            column["name"] = name.strip()
        if "wip_limit" in data:
            wip_limit = data["wip_limit"]
            if wip_limit is not None and (
                not _is_int(wip_limit) or wip_limit < 0
            ):
                return _error(
                    "wip_limit must be a non-negative integer or null", 400
                )
            if wip_limit is not None and len(column["cards"]) > wip_limit:
                return _error(
                    f"wip_limit {wip_limit} is below the current card "
                    f"count ({len(column['cards'])})",
                    409,
                )
            column["wip_limit"] = wip_limit
        return jsonify(_public_column(column))

    @app.delete("/boards/<int:board_id>/columns/<int:column_id>")
    def delete_column(board_id, column_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        column, err = _get_column(board, column_id)
        if err is not None:
            return err
        if column["cards"]:
            return _error(
                f"column {column_id} still holds "
                f"{len(column['cards'])} card(s); move or delete them first",
                409,
            )
        board["columns"] = [c for c in board["columns"] if c["id"] != column_id]
        return jsonify({"deleted": column_id})

    @app.post("/boards/<int:board_id>/columns/<int:column_id>/cards")
    def create_card(board_id, column_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        column, err = _get_column(board, column_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        description = data.get("description", "")
        if not isinstance(description, str):
            return _error("description must be a string", 400)
        assignee = data.get("assignee")
        if assignee is not None and (
            not isinstance(assignee, str) or not assignee.strip()
        ):
            return _error("assignee must be a non-empty string or null", 400)
        priority = data.get("priority", "medium")
        if priority not in PRIORITIES:
            return _error(
                f"priority must be one of {', '.join(PRIORITIES)}", 400
            )
        if column["wip_limit"] is not None and \
                len(column["cards"]) >= column["wip_limit"]:
            return _error(
                f"column {column_id} is at its WIP limit "
                f"({column['wip_limit']})",
                409,
            )
        card_id = app.config["NEXT_CARD_ID"]
        app.config["NEXT_CARD_ID"] += 1
        card = {
            "id": card_id,
            "title": title.strip(),
            "description": description,
            "assignee": assignee.strip() if assignee else None,
            "priority": priority,
            "column_id": column_id,
            "position": len(column["cards"]),
        }
        column["cards"].append(card)
        return jsonify(_public_card(card)), 201

    @app.get("/boards/<int:board_id>/cards/<int:card_id>")
    def get_card(board_id, card_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        card, _column = _find_card(board, card_id)
        if card is None:
            return _error(f"card {card_id} not found in board {board_id}", 404)
        return jsonify(_public_card(card))

    @app.put("/boards/<int:board_id>/cards/<int:card_id>")
    def update_card(board_id, card_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        card, _column = _find_card(board, card_id)
        if card is None:
            return _error(f"card {card_id} not found in board {board_id}", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("title", "description", "assignee", "priority")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide title, description, assignee "
                "and/or priority",
                400,
            )
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            card["title"] = data["title"].strip()
        if "description" in data:
            if not isinstance(data["description"], str):
                return _error("description must be a string", 400)
            card["description"] = data["description"]
        if "assignee" in data:
            assignee = data["assignee"]
            if assignee is not None and (
                not isinstance(assignee, str) or not assignee.strip()
            ):
                return _error("assignee must be a non-empty string or null", 400)
            card["assignee"] = assignee.strip() if assignee else None
        if "priority" in data:
            if data["priority"] not in PRIORITIES:
                return _error(
                    f"priority must be one of {', '.join(PRIORITIES)}", 400
                )
            card["priority"] = data["priority"]
        return jsonify(_public_card(card))

    @app.delete("/boards/<int:board_id>/cards/<int:card_id>")
    def delete_card(board_id, card_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        card, column = _find_card(board, card_id)
        if card is None:
            return _error(f"card {card_id} not found in board {board_id}", 404)
        column["cards"] = [c for c in column["cards"] if c["id"] != card_id]
        _renumber(column)
        return jsonify({"deleted": card_id})

    @app.post("/boards/<int:board_id>/cards/<int:card_id>/move")
    def move_card(board_id, card_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        card, source = _find_card(board, card_id)
        if card is None:
            return _error(f"card {card_id} not found in board {board_id}", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        target_id = data.get("column_id")
        if not _is_int(target_id):
            return _error("column_id is required and must be an integer", 400)
        target, err = _get_column(board, target_id)
        if err is not None:
            return err
        position = data.get("position")
        if position is not None and (
            not _is_int(position) or position < 0
        ):
            return _error("position must be a non-negative integer", 400)
        if target is not source and target["wip_limit"] is not None and \
                len(target["cards"]) >= target["wip_limit"]:
            return _error(
                f"column {target_id} is at its WIP limit "
                f"({target['wip_limit']})",
                409,
            )
        source["cards"] = [c for c in source["cards"] if c["id"] != card_id]
        _renumber(source)
        if position is None or position > len(target["cards"]):
            position = len(target["cards"])
        target["cards"].insert(position, card)
        card["column_id"] = target_id
        _renumber(target)
        return jsonify(_public_card(card))

    @app.get("/boards/<int:board_id>/stats")
    def board_stats(board_id):
        board, err = _get_board(board_id)
        if err is not None:
            return err
        columns = []
        total = 0
        for column in board["columns"]:
            count = len(column["cards"])
            total += count
            columns.append(
                {
                    "column_id": column["id"],
                    "name": column["name"],
                    "cards": count,
                    "wip_limit": column["wip_limit"],
                    "wip_used_pct": (
                        round(count / column["wip_limit"] * 100, 1)
                        if column["wip_limit"] else None
                    ),
                }
            )
        return jsonify(
            {
                "board_id": board_id,
                "total_cards": total,
                "columns": columns,
                "by_priority": {
                    p: sum(
                        1
                        for column in board["columns"]
                        for c in column["cards"]
                        if c["priority"] == p
                    )
                    for p in PRIORITIES
                },
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
