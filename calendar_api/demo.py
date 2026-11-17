"""Demo the calendar_api: create events, list upcoming, filter by date."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calendar_api.app import create_app

client = create_app().test_client()

events = [
    {"title": "Team standup", "start": "2026-11-17T09:00",
     "end": "2026-11-17T09:30", "location": "Room 4"},
    {"title": "Project demo", "start": "2026-11-17T14:00",
     "end": "2026-11-17T15:00", "location": "Hall B"},
    {"title": "Dentist", "start": "2026-11-18T11:00",
     "end": "2026-11-18T11:45"},
    {"title": "Old meeting", "start": "2026-11-16T10:00",
     "end": "2026-11-16T11:00"},
]
for e in events:
    resp = client.post("/events", json=e)
    created = resp.get_json()
    print(f"Created event {created['id']}: {created['title']} "
          f"{created['start']}–{created['end']} -> {resp.status_code}")

all_events = client.get("/events").get_json()
print(f"All events: {len(all_events)} (sorted by start)")

upcoming = client.get(
    "/events/upcoming?now=2026-11-17T00:00").get_json()
print(f"Upcoming from 2026-11-17: {len(upcoming)} -> "
      f"{[e['title'] for e in upcoming]}")

today = client.get("/events?date=2026-11-17").get_json()
print(f"Events on 2026-11-17: {len(today)} -> {[e['title'] for e in today]}")

bad = client.post("/events", json={
    "title": "Backwards", "start": "2026-11-17T10:00",
    "end": "2026-11-17T09:00"})
print(f"End before start -> {bad.status_code}: {bad.get_json()['error']}")
missing = client.get("/events/99")
print(f"Unknown event -> {missing.status_code}: {missing.get_json()['error']}")
