"""Demo: add books and members, vote, schedule a meeting, RSVP."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from book_club_api.app import create_app


def main():
    client = create_app().test_client()

    book1 = client.post(
        "/books", json={"title": "Dune", "author": "Frank Herbert"}
    ).get_json()
    book2 = client.post(
        "/books", json={"title": "Neuromancer", "author": "William Gibson"}
    ).get_json()
    print(f"Added books: {book1['title']} and {book2['title']}")

    member1 = client.post("/members", json={"name": "Ayesha"}).get_json()
    member2 = client.post("/members", json={"name": "Bilal"}).get_json()
    print(f"Members: {member1}, {member2}")

    client.post(f"/books/{book1['id']}/vote", json={"member_id": member1["id"]})
    client.post(f"/books/{book1['id']}/vote", json={"member_id": member2["id"]})
    client.post(f"/books/{book2['id']}/vote", json={"member_id": member1["id"]})
    leaderboard = client.get("/votes").get_json()
    print(f"Vote leaderboard: {[(b['title'], b['votes']) for b in leaderboard]}")

    meeting = client.post(
        "/meetings",
        json={"book_id": book1["id"], "date": "2026-12-05", "location": "Library"},
    ).get_json()
    client.post(
        f"/meetings/{meeting['id']}/rsvp",
        json={"member_id": member1["id"], "attending": True},
    )
    client.post(
        f"/meetings/{meeting['id']}/rsvp",
        json={"member_id": member2["id"], "attending": False},
    )
    final = client.get("/meetings").get_json()
    print(f"Meeting with RSVPs: {final}")


if __name__ == "__main__":
    main()
