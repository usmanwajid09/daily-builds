"""Demo: create a goal, add milestones, and track progress via test client."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goal_tracker_api.app import create_app


def main():
    client = create_app().test_client()

    goal = client.post(
        "/goals",
        json={"title": "Read 12 books", "target": 12, "deadline": "2026-12-31"},
    ).get_json()
    print(f"Created goal: {goal}")

    client.post(f"/goals/{goal['id']}/milestones", json={"title": "Finish book 1"})
    client.post(f"/goals/{goal['id']}/milestones", json={"title": "Finish book 2"})
    with_milestones = client.get(f"/goals/{goal['id']}").get_json()
    print(f"Goal with 2 milestones, progress: {with_milestones['progress']}%")

    done = client.put(
        f"/goals/{goal['id']}/milestones/1", json={"done": True}
    ).get_json()
    print(f"After completing 1 of 2 milestones, progress: {done['progress']}%")

    client.put(f"/goals/{goal['id']}/progress", json={"progress": 75})
    final = client.get(f"/goals/{goal['id']}").get_json()
    print(f"Final goal (milestone progress wins over stored value): {final}")


if __name__ == "__main__":
    main()
