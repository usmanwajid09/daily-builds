"""Demo: create a poll, cast votes, and view results via test_client."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poll_api.app import create_app


def main():
    client = create_app().test_client()

    created = client.post(
        "/polls",
        json={
            "question": "Best language for a quick API?",
            "options": ["Python", "Go", "JavaScript"],
        },
    ).get_json()
    print(f"Created poll: {created}")
    poll_id = created["id"]

    client.post(f"/polls/{poll_id}/vote", json={"voter_id": "alice", "option": "Python"})
    client.post(f"/polls/{poll_id}/vote", json={"voter_id": "bob", "option": "Go"})
    client.post(f"/polls/{poll_id}/vote", json={"voter_id": "carol", "option": "Python"})

    duplicate = client.post(
        f"/polls/{poll_id}/vote", json={"voter_id": "alice", "option": "Go"}
    )
    print(f"Second vote from alice -> {duplicate.status_code}: {duplicate.get_json()}")

    results = client.get(f"/polls/{poll_id}/results").get_json()
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
