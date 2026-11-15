# poll_api

A small Flask REST API for creating polls and voting in them. Each
voter (identified by `voter_id`) may vote only once per poll. Results
report counts and percentages. All state lives in memory; nothing is
written to disk.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/polls` | List all polls with current vote counts. |
| POST | `/polls` | Create a poll: `{"question": str, "options": [str, str, ...]}` (at least 2 unique options). |
| GET | `/polls/<id>` | Fetch one poll with its current results. |
| POST | `/polls/<id>/vote` | Vote: `{"voter_id": str, "option": str}`. A second vote from the same voter returns `409`. |
| GET | `/polls/<id>/results` | Counts and percentages per option. |
| DELETE | `/polls/<id>` | Delete a poll. |

Invalid input returns `400` with a JSON `{"error": ...}` message;
unknown poll ids return `404`.

## Run

```bash
python poll_api/demo.py
python -m pytest tests/test_poll_api.py -q
```

## Example

```
Created poll: {'id': 1, 'question': 'Best language for a quick API?', 'options': ['Python', 'Go', 'JavaScript'], 'votes': {'Python': 0, 'Go': 0, 'JavaScript': 0}, 'total_votes': 0}
Results: {'id': 1, 'question': 'Best language for a quick API?', 'total_votes': 3, 'results': [{'option': 'Python', 'votes': 2, 'percentage': 66.7}, {'option': 'Go', 'votes': 1, 'percentage': 33.3}, {'option': 'JavaScript', 'votes': 0, 'percentage': 0.0}]}
```
