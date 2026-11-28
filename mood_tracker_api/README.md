# mood_tracker_api

A Flask mood tracker API with an in-memory store (nothing is written to
disk). Each app instance starts with an empty store. All responses are
JSON. One entry per date.

An entry is `{"date": "YYYY-MM-DD", "mood": 1-5, "note": "...", "tags": ["..."]}`.
`note` and `tags` are optional on create (default `""` and `[]`).

## Endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/entries` | Create an entry — 201 on success, 409 on duplicate date, 400 on invalid fields |
| GET | `/entries` | List all entries, sorted by date ascending |
| GET | `/entries/<date>` | Fetch one entry — 404 if no entry for that date |
| PUT | `/entries/<date>` | Update mood/note/tags — 404 if unknown, 400 on invalid fields |
| DELETE | `/entries/<date>` | Delete an entry — 404 if unknown |
| GET | `/stats/weekly` | Stats for the last 7 days: entries, average/min/max mood |
| GET | `/stats/trend?days=14` | Daily moods plus first/second half averages and direction (`improving` / `declining` / `stable`) |
| GET | `/stats/tags` | Tag counts, most frequent first |

Errors are JSON like `{"error": "message"}` with the matching
400 / 404 / 409 status code.

## Run the demo

```bash
python mood_tracker_api/demo.py
```

## Example output

```
Created entry (201): {'date': '2026-11-25', 'mood': 2, 'note': 'Rough start to the week', 'tags': ['work', 'tired']}
Created entry (201): {'date': '2026-11-26', 'mood': 3, 'note': 'Steady day', 'tags': ['work']}
Created entry (201): {'date': '2026-11-27', 'mood': 4, 'note': 'Good run and early night', 'tags': ['exercise', 'sleep']}
Created entry (201): {'date': '2026-11-28', 'mood': 5, 'note': 'Great day, shipped the feature', 'tags': ['work', 'exercise']}
Duplicate date rejected (409): {'error': 'an entry already exists for 2026-11-28'}
All entries: [{'date': '2026-11-25', 'mood': 2, 'note': 'Rough start to the week', 'tags': ['work', 'tired']}, {'date': '2026-11-26', 'mood': 3, 'note': 'Steady day', 'tags': ['work']}, {'date': '2026-11-27', 'mood': 4, 'note': 'Good run and early night', 'tags': ['exercise', 'sleep']}, {'date': '2026-11-28', 'mood': 5, 'note': 'Great day, shipped the feature', 'tags': ['work', 'exercise']}]
Weekly stats: {'average_mood': 3.5, 'days': 7, 'entries': 4, 'max_mood': 5, 'min_mood': 2}
Trend (14 days): {'daily': [{'date': '2026-11-25', 'mood': 2}, {'date': '2026-11-26', 'mood': 3}, {'date': '2026-11-27', 'mood': 4}, {'date': '2026-11-28', 'mood': 5}], 'days': 14, 'direction': 'stable', 'first_half_average': None, 'second_half_average': 3.5}
Tag counts: [{'count': 3, 'tag': 'work'}, {'count': 2, 'tag': 'exercise'}, {'count': 1, 'tag': 'sleep'}, {'count': 1, 'tag': 'tired'}]
```

## Tests

```bash
python -m pytest tests/test_mood_tracker_api.py -v
```

38 tests, all passing.
