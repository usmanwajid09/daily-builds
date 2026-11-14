"""Flask flashcard API with a simple spaced-review study mode.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /decks`` — create a deck (``{"name": str}``)
- ``GET /decks`` — list decks with card counts
- ``GET /decks/<id>`` — fetch one deck
- ``DELETE /decks/<id>`` — delete a deck and its cards
- ``POST /decks/<id>/cards`` — add a card (``{"front": str, "back": str}``)
- ``GET /decks/<id>/cards`` — list a deck's cards
- ``DELETE /decks/<id>/cards/<card_id>`` — delete a card
- ``GET /decks/<id>/study`` — the next due card (see below)
- ``POST /decks/<id>/cards/<card_id>/review`` — ``{"known": bool}``
  records a review and updates the card's counts

Study order (simple spaced review): cards never reviewed come first
(oldest first), then the cards with the fewest ``known`` reviews, then
the least recently reviewed. Reviewing a card marks it ``known`` or
``unknown`` and stamps ``last_reviewed``.
"""
from datetime import datetime, timezone

from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory flashcard store."""
    app = Flask(__name__)
    app.config["DECKS"] = {}
    app.config["NEXT_DECK_ID"] = 1
    app.config["NEXT_CARD_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_deck(deck_id):
        deck = app.config["DECKS"].get(deck_id)
        if deck is None:
            return None, _error(f"deck {deck_id} not found", 404)
        return deck, None

    def _get_card(deck, card_id):
        card = deck["cards"].get(card_id)
        if card is None:
            return None, _error(f"card {card_id} not found in deck {deck['id']}", 404)
        return card, None

    def _public_deck(deck):
        return {"id": deck["id"], "name": deck["name"], "card_count": len(deck["cards"])}

    def _public_card(card):
        return dict(card)

    @app.post("/decks")
    def create_deck():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        deck_id = app.config["NEXT_DECK_ID"]
        app.config["NEXT_DECK_ID"] += 1
        deck = {"id": deck_id, "name": name.strip(), "cards": {}}
        app.config["DECKS"][deck_id] = deck
        return jsonify(_public_deck(deck)), 201

    @app.get("/decks")
    def list_decks():
        decks = sorted(app.config["DECKS"].values(), key=lambda d: d["id"])
        return jsonify([_public_deck(d) for d in decks])

    @app.get("/decks/<int:deck_id>")
    def get_deck(deck_id):
        deck, err = _get_deck(deck_id)
        if err is not None:
            return err
        return jsonify(_public_deck(deck))

    @app.delete("/decks/<int:deck_id>")
    def delete_deck(deck_id):
        _deck, err = _get_deck(deck_id)
        if err is not None:
            return err
        del app.config["DECKS"][deck_id]
        return jsonify({"deleted": deck_id})

    @app.post("/decks/<int:deck_id>/cards")
    def add_card(deck_id):
        deck, err = _get_deck(deck_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        front = data.get("front")
        back = data.get("back")
        if not isinstance(front, str) or not front.strip():
            return _error("front is required and must be a non-empty string", 400)
        if not isinstance(back, str) or not back.strip():
            return _error("back is required and must be a non-empty string", 400)
        card_id = app.config["NEXT_CARD_ID"]
        app.config["NEXT_CARD_ID"] += 1
        card = {
            "id": card_id,
            "deck_id": deck_id,
            "front": front.strip(),
            "back": back.strip(),
            "known_count": 0,
            "unknown_count": 0,
            "last_reviewed": None,
        }
        deck["cards"][card_id] = card
        return jsonify(_public_card(card)), 201

    @app.get("/decks/<int:deck_id>/cards")
    def list_cards(deck_id):
        deck, err = _get_deck(deck_id)
        if err is not None:
            return err
        cards = sorted(deck["cards"].values(), key=lambda c: c["id"])
        return jsonify([_public_card(c) for c in cards])

    @app.delete("/decks/<int:deck_id>/cards/<int:card_id>")
    def delete_card(deck_id, card_id):
        deck, err = _get_deck(deck_id)
        if err is not None:
            return err
        _card, err = _get_card(deck, card_id)
        if err is not None:
            return err
        del deck["cards"][card_id]
        return jsonify({"deleted": card_id})

    @app.get("/decks/<int:deck_id>/study")
    def next_study_card(deck_id):
        deck, err = _get_deck(deck_id)
        if err is not None:
            return err
        cards = list(deck["cards"].values())
        if not cards:
            return _error(f"deck {deck_id} has no cards to study", 404)

        def sort_key(card):
            never_reviewed = 0 if card["last_reviewed"] is None else 1
            return (
                never_reviewed,
                card["known_count"],
                card["last_reviewed"] or "",
                card["id"],
            )

        return jsonify(_public_card(sorted(cards, key=sort_key)[0]))

    @app.post("/decks/<int:deck_id>/cards/<int:card_id>/review")
    def review_card(deck_id, card_id):
        deck, err = _get_deck(deck_id)
        if err is not None:
            return err
        card, err = _get_card(deck, card_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        known = data.get("known")
        if not isinstance(known, bool):
            return _error("known is required and must be a boolean", 400)
        if known:
            card["known_count"] += 1
        else:
            card["unknown_count"] += 1
        card["last_reviewed"] = datetime.now(timezone.utc).isoformat()
        return jsonify(_public_card(card))

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
