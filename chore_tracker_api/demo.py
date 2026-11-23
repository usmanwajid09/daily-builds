"""Demo: create a chore, mark it done twice, watch the rotation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chore_tracker_api.app import create_app


def main():
    client = create_app().test_client()

    chore = client.post(
        "/chores",
        json={
            "name": "Wash dishes",
            "assignees": ["Usman", "Ali", "Sara"],
            "frequency_days": 2,
        },
    ).get_json()
    print(f"Created chore: {chore}")

    done1 = client.post(f"/chores/{chore['id']}/done", json={}).get_json()
    print(
        f"After 1st done: current assignee is {done1['current_assignee']}, "
        f"times done {done1['times_done']}"
    )
    done2 = client.post(f"/chores/{chore['id']}/done", json={}).get_json()
    print(
        f"After 2nd done: current assignee is {done2['current_assignee']}, "
        f"due {done2['due_date']}, overdue {done2['overdue']}"
    )

    stale = client.post(
        "/chores",
        json={"name": "Vacuum", "assignees": ["Usman"], "frequency_days": 1},
    ).get_json()
    client.post(f"/chores/{stale['id']}/done", json={"date": "2026-11-10"})
    flagged = client.get(f"/chores/{stale['id']}").get_json()
    print(f"Chore done 13 days ago, every 1 day: overdue = {flagged['overdue']}")


if __name__ == "__main__":
    main()
