# Travel Planner API

A tiny REST travel planner API built with Flask. Trips, itinerary days
and activities live in an in-memory store (fresh on every app start —
nothing is written to disk), so it is ideal for learning REST patterns
and for testing with Flask's test client.

## Why

A travel planner is a small nested CRUD app: a trip has itinerary days,
each day has activities with a cost, and the cost endpoint rolls the
activities up into a per-day breakdown and a per-trip total.

## Usage

```bash
pip install -r requirements.txt
python travel_planner_api/demo.py
```

The demo creates a Hunza trip, adds two itinerary days with four
activities, prints the cost breakdown, and shows a day outside the
trip's dates being rejected.

To run the app for real:

```bash
flask --app "travel_planner_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/trips` | Create a trip: `{"name", "destination", "start_date", "end_date"}` |
| GET | `/trips` | List all trips (with day/activity counts and total cost) |
| GET | `/trips/<id>` | Fetch one trip |
| DELETE | `/trips/<id>` | Delete a trip (and its days and activities) |
| POST | `/trips/<id>/days` | Add an itinerary day: `{"date", "title"}` — date must be inside the trip |
| GET | `/trips/<id>/days` | List a trip's days, earliest first |
| DELETE | `/trips/<id>/days/<day_id>` | Delete a day (and its activities) |
| POST | `/trips/<id>/days/<day_id>/activities` | Add an activity: `{"name", "cost", "description"?}` |
| GET | `/trips/<id>/days/<day_id>/activities` | List a day's activities |
| GET | `/trips/<id>/cost` | Per-trip total cost with a per-day breakdown |

A trip looks like
`{"id": 1, "name": "Northern Areas", "destination": "Hunza", "start_date": "2026-12-10", "end_date": "2026-12-12", "day_count": 2, "activity_count": 4, "total_cost": 19500}`.
Validation errors return `400` with `{"error": "..."}`; unknown ids
return `404`; a duplicate day for the same date returns `409`.

## Run tests

```bash
python -m pytest tests/test_travel_planner_api.py -v
```
