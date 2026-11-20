"""Demo: add books, search by author, log reading, finish one, show stats."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reading_list_api.app import create_app


def main():
    client = create_app().test_client()

    dune = client.post(
        "/books",
        json={
            "title": "Dune",
            "author": "Frank Herbert",
            "total_pages": 688,
            "status": "reading",
            "pages_read": 100,
        },
    ).get_json()
    print(f"Added: {dune['title']!r} by {dune['author']} "
          f"({dune['pages_read']}/{dune['total_pages']} pages, "
          f"{dune['progress_pct']}% read)")

    foundation = client.post(
        "/books",
        json={
            "title": "Foundation",
            "author": "Isaac Asimov",
            "total_pages": 296,
        },
    ).get_json()
    print(f"Added: {foundation['title']!r} by {foundation['author']} "
          f"({foundation['status']})")

    herbert = client.get("/books", query_string={"author": "herbert"}).get_json()
    print(f"Search author 'herbert': {[b['title'] for b in herbert]}")

    after = client.post(f"/books/{dune['id']}/read", json={"pages": 200}).get_json()
    print(f"Read 200 more pages of Dune: {after['pages_read']}/"
          f"{after['total_pages']} ({after['progress_pct']}%, "
          f"status {after['status']})")

    done = client.post(f"/books/{foundation['id']}/finish").get_json()
    print(f"Finished {done['title']!r}: {done['pages_read']}/"
          f"{done['total_pages']} pages, status {done['status']}")

    stats = client.get("/stats").get_json()
    print(f"Stats: {stats['total_books']} books, "
          f"{stats['pages_read']}/{stats['total_pages']} pages read, "
          f"overall progress {stats['overall_progress_pct']}%, "
          f"by status {stats['by_status']}")

    bad = client.post("/books", json={"title": "No Author", "total_pages": 10})
    print(f"Missing author rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
