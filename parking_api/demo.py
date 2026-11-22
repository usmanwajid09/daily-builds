"""Demo: create spots, check a car in and out, and read the occupancy summary."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parking_api.app import create_app


def main():
    client = create_app().test_client()

    for number, zone, rate in (("A1", "Ground", 100), ("A2", "Ground", 100),
                               ("B1", "Basement", 80)):
        spot = client.post(
            "/spots",
            json={"spot_number": number, "zone": zone, "hourly_rate": rate},
        ).get_json()
        print(
            f"Created spot #{spot['id']}: {spot['spot_number']} "
            f"(zone {spot['zone']}, Rs {spot['hourly_rate']}/hour) — {spot['status']}"
        )

    session = client.post(
        "/spots/1/check-in",
        json={
            "license_plate": "LEA-123",
            "started_at": "2026-11-22T08:00:00+00:00",
        },
    ).get_json()
    print(
        f"Check-in: {session['license_plate']} into spot #1 "
        f"at {session['started_at']}"
    )

    double = client.post(
        "/spots/1/check-in", json={"license_plate": "LEA-999"}
    )
    print(f"Second check-in on occupied spot #1: {double.status_code} "
          f"({double.get_json()['error']})")

    summary = client.get("/summary").get_json()
    print(
        f"Summary: {summary['occupied_spots']}/{summary['total_spots']} occupied "
        f"({summary['occupancy_percent']}%), revenue Rs {summary['total_revenue']}"
    )

    done = client.post(
        "/spots/1/check-out",
        json={"ended_at": "2026-11-22T10:30:00+00:00"},
    ).get_json()
    print(
        f"Check-out: {done['license_plate']} parked "
        f"{done['duration_minutes']} minutes, billed "
        f"{done['hours_billed']} hour(s), fee Rs {done['fee']}"
    )

    summary = client.get("/summary").get_json()
    print(
        f"Summary: {summary['occupied_spots']}/{summary['total_spots']} occupied "
        f"({summary['occupancy_percent']}%), revenue Rs {summary['total_revenue']}"
    )


if __name__ == "__main__":
    main()
