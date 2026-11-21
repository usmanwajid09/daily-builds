# kanban_api

Flask kanban board API: boards, columns, cards, moving cards between
columns, and WIP (work-in-progress) limits. In-memory store, fresh
for each app instance.

## Endpoints

- `POST /boards` — create a board: `{"name": str}`.
- `GET /boards` — list boards.
- `GET /boards/<id>` — fetch one board with its columns and cards.
- `DELETE /boards/<id>` — delete a board and everything in it.
- `POST /boards/<id>/columns` — add a column: `{"name": str,
  "wip_limit": int | null?}` (null = unlimited). Duplicate column
  names on a board are rejected (409).
- `PUT /boards/<id>/columns/<col_id>` — rename a column and/or change
  its WIP limit (409 if the new limit is below the current card
  count).
- `DELETE /boards/<id>/columns/<col_id>` — delete an empty column
  (409 if it still holds cards).
- `POST /boards/<id>/columns/<col_id>/cards` — add a card:
  `{"title": str, "description": str?, "assignee": str?,
  "priority": "low" | "medium" | "high"?}` (409 if the column is at
  its WIP limit).
- `GET /boards/<id>/cards/<card_id>` — fetch one card.
- `PUT /boards/<id>/cards/<card_id>` — update title, description,
  assignee and/or priority.
- `DELETE /boards/<id>/cards/<card_id>` — delete a card.
- `POST /boards/<id>/cards/<card_id>/move` — move a card:
  `{"column_id": int, "position": int?}` (409 if the target column
  is at its WIP limit; positions are 0-based and renumbered).
- `GET /boards/<id>/stats` — card counts per column, WIP usage %,
  total cards, counts by priority.

Bad input returns 400, missing boards/columns/cards 404, WIP and
name conflicts 409.

## Run

```bash
python -m pytest tests/test_kanban_api.py -q
python kanban_api/demo.py
```

## Example output

```
Created board #1: 'Website Redesign'
Columns: To Do (#1), Doing (#2, WIP 2), Done (#3)
Added card #1: 'Design homepage' (high, position 0)
Added card #2: 'Write copy' (medium, position 1)
Added card #3: 'Pick fonts' (low, position 2)
Moved 'Design homepage' to Doing: column 2, position 0
Moved 'Write copy' to Doing (WIP now 2/2)
Third card into Doing rejected (WIP limit): 409 {'error': 'column 2 is at its WIP limit (2)'}
Moved 'Design homepage' to Done at position 0
Stats: 3 cards, by priority {'high': 1, 'low': 1, 'medium': 1}
  To Do: 1 card(s), WIP None, used None%
  Doing: 1 card(s), WIP 2, used 50.0%
  Done: 1 card(s), WIP None, used None%
```
