# Flashcard API

A tiny Flask flashcard API with decks, cards, and a simple spaced-review
study mode. Everything lives in an in-memory store (fresh on every app
start — nothing is written to disk).

## Why

Flashcards are a small domain with real structure: decks contain cards,
cards carry review history, and "what should I study next?" is a
genuine ordering problem. This app shows nested resources
(`/decks/<id>/cards/...`) and a derived ordering rule.

## Study order

`GET /decks/<id>/study` returns the next due card: cards never reviewed
come first (oldest first), then the cards with the fewest `known`
reviews, then the least recently reviewed. Reviewing a card with
`{"known": true}` or `{"known": false}` bumps its known/unknown count
and stamps `last_reviewed`.

## Usage

```bash
pip install -r requirements.txt
python -m flashcard_api.demo
```

The demo builds a Python basics deck, studies all three cards, and
shows the next due card.

To run the app for real:

```bash
flask --app "flashcard_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/decks` | Create a deck: `{"name": str}` |
| GET | `/decks` | List decks with card counts |
| GET | `/decks/<id>` | Fetch one deck |
| DELETE | `/decks/<id>` | Delete a deck and its cards |
| POST | `/decks/<id>/cards` | Add a card: `{"front": str, "back": str}` |
| GET | `/decks/<id>/cards` | List a deck's cards |
| DELETE | `/decks/<id>/cards/<card_id>` | Delete a card |
| GET | `/decks/<id>/study` | Next due card (404 if the deck is empty) |
| POST | `/decks/<id>/cards/<card_id>/review` | Record a review: `{"known": bool}` |

A card looks like
`{"id": 1, "deck_id": 1, "front": "What does len() do?",
"back": "Returns the number of items...", "known_count": 0,
"unknown_count": 0, "last_reviewed": null}`.

Validation errors return `400` with `{"error": "..."}`; unknown deck
or card ids and routes return `404` with `{"error": "..."}`.

## Run tests

```bash
python -m pytest tests/test_flashcard_api.py -v
```
