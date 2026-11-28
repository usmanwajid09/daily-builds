"""Demo: run a fundraiser campaign, take donations, show progress and top donors."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fundraiser_api.app import create_app


def main():
    client = create_app().test_client()

    campaign = client.post("/campaigns", json={
        "title": "School Library Fund",
        "goal_amount": 1000,
        "description": "New books for the school library",
    }).get_json()
    print(f"Created campaign {campaign['id']}: {campaign['title']!r} "
          f"with goal ${campaign['goal_amount']}")

    for donor, amount, message in [
        ("Ali", 250, "For the kids!"),
        ("Sara", 400, ""),
        ("Ali", 150, "Second gift"),
        ("Bilal", 100, ""),
    ]:
        donation = client.post(
            f"/campaigns/{campaign['id']}/donate",
            json={"donor_name": donor, "amount": amount, "message": message},
        ).get_json()
        print(f"Donation {donation['id']}: {donor} gave ${amount}")

    progress = client.get(f"/campaigns/{campaign['id']}/progress").get_json()
    print(f"Progress: raised ${progress['raised']} of ${progress['goal_amount']} "
          f"({progress['progress_percent']}%), ${progress['remaining']} remaining, "
          f"{progress['donor_count']} donors, goal reached: {progress['goal_reached']}")

    top = client.get(f"/campaigns/{campaign['id']}/top-donors").get_json()
    print("Top donors: " + ", ".join(
        f"{d['donor_name']} (${d['total_donated']}, {d['donation_count']} donations)"
        for d in top
    ))

    bad = client.post(f"/campaigns/{campaign['id']}/donate",
                      json={"donor_name": "Zara", "amount": -50})
    print(f"Invalid donation rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
