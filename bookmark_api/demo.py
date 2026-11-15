"""Demo: create, search, update, and delete bookmarks via test_client."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bookmark_api.app import create_app


def main():
    client = create_app().test_client()

    created = client.post(
        "/bookmarks",
        json={
            "title": "Flask docs",
            "url": "https://flask.palletsprojects.com/",
            "tags": ["python", "web"],
        },
    ).get_json()
    print(f"Created bookmark: {created}")
    client.post(
        "/bookmarks",
        json={
            "title": "Real Python",
            "url": "https://realpython.com/",
            "tags": ["python", "tutorials"],
        },
    )

    tagged = client.get("/bookmarks?tag=python").get_json()
    print(f"Bookmarks tagged 'python': {len(tagged)}")
    found = client.get("/bookmarks?q=flask").get_json()
    print(f"Search 'flask': {found}")

    updated = client.put(
        f"/bookmarks/{created['id']}",
        json={"tags": ["python", "web", "docs"]},
    ).get_json()
    print(f"Updated bookmark: {updated}")

    deleted = client.delete(f"/bookmarks/{created['id']}").get_json()
    print(f"Deleted: {deleted}")
    remaining = client.get("/bookmarks").get_json()
    print(f"Bookmarks left: {remaining}")


if __name__ == "__main__":
    main()
