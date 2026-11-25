"""Demo: create a plan, add members, check in, and show the expiry flag on an old membership."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_membership_api.app import create_app


def main():
    client = create_app().test_client()

    monthly = client.post("/plans", json={
        "name": "Monthly", "duration_days": 30, "price": 5000,
    }).get_json()
    print(f"Created plan: {monthly}")

    usman = client.post("/members", json={
        "name": "Usman", "plan_id": monthly["id"], "start_date": "2026-11-25",
    }).get_json()
    old = client.post("/members", json={
        "name": "Ali", "plan_id": monthly["id"], "start_date": "2026-09-01",
    }).get_json()
    print(f"Members: {usman}, {old}")

    checkin = client.post(f"/members/{usman['id']}/check-in", json={"date": "2026-11-25"}).get_json()
    print(f"Check-in: {checkin}")

    duplicate = client.post(f"/members/{usman['id']}/check-in", json={"date": "2026-11-25"})
    print(f"Same-day repeat rejected: {duplicate.status_code} {duplicate.get_json()}")

    expired = client.post(f"/members/{old['id']}/check-in", json={"date": "2026-11-25"})
    print(f"Expired member check-in rejected: {expired.status_code} {expired.get_json()}")

    expired_list = client.get("/members/expired").get_json()
    print(f"Expired members: {[m['name'] for m in expired_list]}")
    print(f"Headline: {len(expired_list)} expired member(s), "
          f"Usman has {usman['days_remaining']} days remaining")


if __name__ == "__main__":
    main()
