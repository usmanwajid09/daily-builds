"""Demo: build a deck, study cards, and review them."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flashcard_api.app import create_app


def main():
    client = create_app().test_client()

    deck = client.post("/decks", json={"name": "Python basics"}).get_json()
    print(f"Created deck: {deck}")

    cards = [
        ("What does len() do?", "Returns the number of items in a container."),
        ("What is a list comprehension?", "A compact way to build a list."),
        ("What does dict.get() return?", "The value, or a default if the key is missing."),
    ]
    for front, back in cards:
        created = client.post(f"/decks/{deck['id']}/cards", json={"front": front, "back": back}).get_json()
        print(f"Added card #{created['id']}: {created['front']}")

    for _ in range(3):
        nxt = client.get(f"/decks/{deck['id']}/study").get_json()
        reviewed = client.post(
            f"/decks/{deck['id']}/cards/{nxt['id']}/review", json={"known": True}
        ).get_json()
        print(f"Studied: {reviewed['front']!r} -> known_count={reviewed['known_count']}")

    # All three reviewed once: the next due card is the least-known
    # (all tied at 1), so the oldest-reviewed one comes first.
    nxt = client.get(f"/decks/{deck['id']}/study").get_json()
    print(f"Next due card: #{nxt['id']} {nxt['front']!r}")

    bad = client.post("/decks", json={"name": "   "})
    print(f"Blank deck name rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
