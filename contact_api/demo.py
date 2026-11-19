"""Demo: submit, list, fetch and delete contact-form messages."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contact_api.app import create_app


def main():
    client = create_app().test_client()

    first = client.post(
        "/messages",
        json={"name": "Ayesha", "email": "ayesha@example.com", "message": "Love the site!"},
    ).get_json()
    print(f"Created message: {first}")
    bad = client.post(
        "/messages",
        json={"name": "Bob", "email": "not-an-email", "message": "Hi"},
    )
    print(f"Bad email rejected: {bad.status_code} {bad.get_json()}")

    client.post(
        "/messages",
        json={"name": "Bilal", "email": "bilal@example.com", "message": "When is the next batch?"},
    )
    messages = client.get("/messages").get_json()
    print(f"{len(messages)} message(s) stored: {[m['name'] for m in messages]}")

    client.delete(f"/messages/{first['id']}")
    remaining = client.get("/messages").get_json()
    print(f"After delete, {len(remaining)} message(s) remain: {[m['name'] for m in remaining]}")


if __name__ == "__main__":
    main()
