# plant_care_api

A small Flask API for tracking houseplants and their watering schedules.
Everything lives in memory — a fresh, empty store is created for each
app instance and nothing is written to disk. All responses are JSON.

Run the demo from the repo root:

```bash
python plant_care_api/demo.py
```

## Endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/plants` | Create a plant (`name`, optional `species`, `watering_interval_days` > 0, optional `last_watered` defaulting to today). Returns 201. |
| GET | `/plants` | List all plants, each with computed `next_watering` and `days_until_watering` (negative when overdue). |
| GET | `/plants/due` | Plants due today or overdue, sorted by `next_watering` ascending. |
| GET | `/plants/<id>` | Fetch one plant. Unknown id → 404. |
| PUT | `/plants/<id>` | Update `name`, `species`, `watering_interval_days` or `last_watered`. Unknown id → 404, invalid input → 400. |
| DELETE | `/plants/<id>` | Delete a plant. Unknown id → 404. |
| POST | `/plants/<id>/water` | Log a watering (`date` optional, defaults to today). Future or malformed dates → 400, unknown plant → 404. |
| GET | `/plants/<id>/water-log` | Watering events, newest first. Unknown plant → 404. |

Errors are JSON, e.g. `{"error": "plant 999 not found"}` with the
appropriate 400 or 404 status code.

## Example output

Output of `python plant_care_api/demo.py` (run 2026-11-28):

```
Created plant: {'days_until_watering': -3, 'id': 1, 'last_watered': '2026-11-18', 'name': 'Monstera', 'next_watering': '2026-11-25', 'species': 'Monstera deliciosa', 'watering_interval_days': 7}
Created plant: {'days_until_watering': 2, 'id': 2, 'last_watered': '2026-11-16', 'name': 'Snake Plant', 'next_watering': '2026-11-30', 'species': 'Sansevieria trifasciata', 'watering_interval_days': 14}
Created plant: {'days_until_watering': 0, 'id': 3, 'last_watered': '2026-11-21', 'name': 'Pothos', 'next_watering': '2026-11-28', 'species': 'Epipremnum aureum', 'watering_interval_days': 7}
Due for watering today or overdue: ['Monstera', 'Pothos']
Watered Monstera: last_watered=2026-11-28, next_watering=2026-12-05
Water log for Monstera (newest first): [{'date': '2026-11-28'}, {'date': '2026-11-18'}]
All plants: [{'days_until_watering': 7, 'id': 1, 'last_watered': '2026-11-28', 'name': 'Monstera', 'next_watering': '2026-12-05', 'species': 'Monstera deliciosa', 'watering_interval_days': 7}, {'days_until_watering': 2, 'id': 2, 'last_watered': '2026-11-16', 'name': 'Snake Plant', 'next_watering': '2026-11-30', 'species': 'Sansevieria trifasciata', 'watering_interval_days': 14}, {'days_until_watering': 0, 'id': 3, 'last_watered': '2026-11-21', 'name': 'Pothos', 'next_watering': '2026-11-28', 'species': 'Epipremnum aureum', 'watering_interval_days': 7}]
```
