"""Demo: create a campaign, take donations, watch progress toward the goal."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from donation_api.app import create_app


def main():
    client = create_app().test_client()

    campaign = client.post(
        "/campaigns",
        json={
            "title": "School Books for Vehari",
            "goal_amount": 100000,
            "organizer": "NovaStack Community",
            "description": "Buy textbooks for 200 students.",
        },
    ).get_json()
    print(
        f"Created campaign #{campaign['id']}: {campaign['title']!r} "
        f"— goal {campaign['goal_amount']}"
    )

    for donor, amount in (("Usman", 25000), ("Hamza", 40000), ("Usman", 10000)):
        donation = client.post(
            f"/campaigns/{campaign['id']}/donations",
            json={"donor_name": donor, "amount": amount},
        ).get_json()
        print(
            f"Donation #{donation['id']}: {donation['donor_name']} gave "
            f"{donation['amount']}"
        )

    summary = client.get(f"/campaigns/{campaign['id']}/summary").get_json()
    print(
        f"Summary: raised {summary['total_raised']} of {summary['goal_amount']} "
        f"({summary['progress_percent']}%), {summary['donation_count']} donations "
        f"from {summary['donor_count']} donors, "
        f"{summary['amount_remaining']} remaining, "
        f"goal reached: {summary['goal_reached']}"
    )

    final = client.post(
        f"/campaigns/{campaign['id']}/donations",
        json={"donor_name": "Anonymous", "amount": 25000},
    ).get_json()
    print(f"Donation #{final['id']}: {final['donor_name']} gave {final['amount']}")

    summary = client.get(f"/campaigns/{campaign['id']}/summary").get_json()
    print(
        f"Summary: raised {summary['total_raised']} of {summary['goal_amount']} "
        f"({summary['progress_percent']}%), goal reached: {summary['goal_reached']}"
    )


if __name__ == "__main__":
    main()
