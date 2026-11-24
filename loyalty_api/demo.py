"""Demo: register members, earn points, climb tiers and claim a reward."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loyalty_api.app import create_app


def main():
    client = create_app().test_client()

    alice = client.post(
        "/members", json={"name": "Alice", "email": "alice@example.com"}
    ).get_json()
    bob = client.post(
        "/members", json={"name": "Bob", "email": "bob@example.com"}
    ).get_json()
    client.post(f"/members/{alice['id']}/earn", json={"points": 1200})
    client.post(f"/members/{bob['id']}/earn", json={"points": 450})
    mug = client.post(
        "/rewards", json={"name": "Free coffee mug", "cost": 500}
    ).get_json()
    claimed = client.post(
        f"/members/{alice['id']}/redeem-reward", json={"reward_id": mug["id"]}
    ).get_json()

    for member in client.get("/leaderboard").get_json():
        print(
            f"{member['name']}: {member['points']} points, tier {member['tier']}"
        )
    print(
        f"Alice claimed {claimed['reward']['name']!r} for "
        f"{claimed['reward']['cost']} points; balance now "
        f"{claimed['member']['points']} points"
    )
    history = client.get(f"/members/{alice['id']}/transactions").get_json()
    print(f"Alice has {len(history)} transactions on record")


if __name__ == "__main__":
    main()
