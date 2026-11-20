"""Demo: create a small event, RSVP until it is full, show attendees and summary."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_rsvp_api.app import create_app


def main():
    client = create_app().test_client()

    event = client.post(
        "/events",
        json={"title": "Board game night", "date": "2026-11-27",
              "capacity": 2, "location": "Usman's place"},
    ).get_json()
    print(f"Created event: {event['title']!r} on {event['date']} "
          f"(capacity {event['capacity']}, {event['spots_remaining']} spots left)")

    for name, response in [("Ali", "yes"), ("Sara", "yes")]:
        rsvp = client.post(f"/events/{event['id']}/rsvp",
                           json={"name": name, "response": response}).get_json()
        print(f"RSVP: {rsvp['name']} -> {rsvp['response']}")

    full = client.post(f"/events/{event['id']}/rsvp",
                       json={"name": "Bilal", "response": "yes"})
    print(f"Third 'yes' refused (event full): {full.status_code} "
          f"{full.get_json()}")

    maybe = client.post(f"/events/{event['id']}/rsvp",
                        json={"name": "Bilal", "response": "maybe"}).get_json()
    print(f"RSVP: {maybe['name']} -> {maybe['response']} (maybes don't take a spot)")

    attendees = client.get(f"/events/{event['id']}/attendees").get_json()
    print(f"Attendees ({len(attendees)}): "
          + ", ".join(f"{a['name']} ({a['response']})" for a in attendees))

    summary = client.get(f"/events/{event['id']}/summary").get_json()
    print(f"Summary: yes={summary['yes']}, no={summary['no']}, "
          f"maybe={summary['maybe']}, spots remaining={summary['spots_remaining']}")

    changed = client.put("/rsvps/2", json={"response": "no"}).get_json()
    print(f"Sara changed to {changed['response']!r}; "
          f"spots now {client.get(f'/events/{event['id']}/summary').get_json()['spots_remaining']}")

    bad = client.post("/events", json={"title": "Bad date", "date": "27-11-2026",
                                       "capacity": 5})
    print(f"Bad date rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
