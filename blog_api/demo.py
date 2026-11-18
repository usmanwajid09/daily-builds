"""Demo: create posts, publish one, filter by tag/status and search."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blog_api.app import create_app


def main():
    client = create_app().test_client()

    first = client.post(
        "/posts",
        json={
            "title": "Hello, Flask!",
            "content": "My first post about building APIs with Flask.",
            "tags": ["flask", "python"],
        },
    ).get_json()
    print(f"Created post: {first['title']!r} -> slug {first['slug']!r} "
          f"({first['status']})")

    dup = client.post(
        "/posts",
        json={"title": "Hello, Flask!", "content": "Same title, second post."},
    ).get_json()
    print(f"Same title again -> unique slug {dup['slug']!r}")

    second = client.post(
        "/posts",
        json={
            "title": "Testing with pytest",
            "content": "How to test a Flask app with its test client.",
            "tags": ["testing", "python"],
            "status": "published",
        },
    ).get_json()
    print(f"Created post: {second['title']!r} -> slug {second['slug']!r} "
          f"({second['status']})")

    published = client.post(f"/posts/{first['slug']}/publish").get_json()
    print(f"Published {published['slug']!r}: status is now {published['status']}")

    all_posts = client.get("/posts").get_json()
    print(f"All posts: {len(all_posts)}")
    live = client.get("/posts", query_string={"status": "published"}).get_json()
    print(f"Published posts: {len(live)}")
    python_posts = client.get("/posts", query_string={"tag": "python"}).get_json()
    print(f"Tagged 'python': {[p['slug'] for p in python_posts]}")
    found = client.get("/posts", query_string={"search": "pytest"}).get_json()
    print(f"Search 'pytest': {[p['slug'] for p in found]}")

    updated = client.put(
        f"/posts/{second['slug']}", json={"title": "Testing Flask with pytest"}
    ).get_json()
    print(f"Updated title to {updated['title']!r}; slug stays {updated['slug']!r}")

    bad = client.post("/posts", json={"title": "!!!", "content": "no letters"})
    print(f"Title with no letters rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
