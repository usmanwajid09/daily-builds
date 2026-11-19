# survey_api

A small Flask survey API: surveys with typed questions, response
submission with validation, and per-question summaries, kept in memory
(a fresh store per `create_app()` call, nothing written to disk).

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/surveys` | Create a survey: `{"title", "questions": [...]}` |
| GET | `/surveys` | List surveys (each with `response_count`) |
| GET | `/surveys/<id>` | Fetch one survey |
| DELETE | `/surveys/<id>` | Delete a survey and its responses |
| POST | `/surveys/<id>/responses` | Submit a response: `{"answers": {"<question_id>": value}}` |
| GET | `/surveys/<id>/responses` | List submitted responses |
| GET | `/surveys/<id>/summary` | Per-question summary |

## Question types

- `choice` — needs at least two `options`; answers must be one of them.
  Summary: a count per option.
- `rating` — answered with an integer 1–5. Summary: count and average.
- `text` — answered with a non-empty string. Summary: the list of answers.

Every question must be answered in each response; a missing, unknown
or out-of-range answer returns JSON `400`. An unknown survey returns
JSON `404`.

## Run

```bash
python survey_api/demo.py
python -m pytest tests/test_survey_api.py -v
```
