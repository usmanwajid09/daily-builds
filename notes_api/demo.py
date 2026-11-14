"""Demo: create, search, update and delete notes via the test client."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notes_api.app import create_app


def main():
    client = create_app().test_client()

    first = client.post(
        "/notes",
        json={"title": "Grocery list", "body": "Milk, eggs, bread", "tags": ["home"]},
    ).get_json()
    print(f"Created note: {first}")
    client.post(
        "/notes",
        json={"title": "Flask tips", "body": "Use create_app factories", "tags": ["code"]},
    )

    found = client.get("/notes", query_string={"search": "milk"}).get_json()
    print(f"Search 'milk' found {len(found)} note(s): {[n['title'] for n in found]}")
    tagged = client.get("/notes", query_string={"tag": "code"}).get_json()
    print(f"Tag 'code' found {len(tagged)} note(s): {[n['title'] for n in tagged]}")

    updated = client.put(
        f"/notes/{first['id']}", json={"body": "Milk, eggs, bread, butter"}
    ).get_json()
    print(f"Updated note body: {updated['body']!r}")

    client.delete(f"/notes/{first['id']}")
    remaining = client.get("/notes").get_json()
    print(f"After delete, {len(remaining)} note(s) remain: {[n['title'] for n in remaining]}")


if __name__ == "__main__":
    main()
