"""Demo: an employee with 20 annual days takes 5 days off, approved."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leave_tracker_api.app import create_app


def main():
    client = create_app().test_client()

    employee = client.post(
        "/employees", json={"name": "Sara", "balances": {"annual": 20}}
    ).get_json()
    print(f"Created employee: {employee['name']} (id {employee['id']}), "
          f"balances {employee['balances']}")

    req = client.post(
        f"/employees/{employee['id']}/leave",
        json={
            "type": "annual",
            "start_date": "2026-12-07",
            "end_date": "2026-12-11",
            "reason": "Winter break",
        },
    ).get_json()
    print(f"Requested {req['days']} annual days "
          f"({req['start_date']} to {req['end_date']}): {req['status']}")

    approved = client.post(f"/leave/{req['id']}/approve").get_json()
    print(f"Request {approved['id']} {approved['status']}")

    balance = client.get(f"/employees/{employee['id']}/balance").get_json()
    print(f"Remaining balances: {balance['balances']} "
          f"(annual {balance['balances']['annual']} of 20 left)")

    overlap = client.post(
        f"/employees/{employee['id']}/leave",
        json={
            "type": "annual",
            "start_date": "2026-12-10",
            "end_date": "2026-12-12",
        },
    )
    print(f"Overlapping request rejected: {overlap.status_code} "
          f"{overlap.get_json()}")
    too_much = client.post(
        f"/employees/{employee['id']}/leave",
        json={
            "type": "annual",
            "start_date": "2027-01-05",
            "end_date": "2027-01-25",
        },
    )
    print(f"Request over balance rejected: {too_much.status_code} "
          f"{too_much.get_json()}")


if __name__ == "__main__":
    main()
