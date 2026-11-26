"""Demo: two threads, posts with upvotes, and a keyword search."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forum_api.app import create_app


def main():
    client = create_app().test_client()

    threads = {}
    for title, author in (
        ("Best Python tips for beginners", "Usman"),
        ("Flask vs Django in 2026", "Ayesha"),
    ):
        created = client.post(
            "/threads", json={"title": title, "author": author}
        ).get_json()
        threads[title] = created["id"]
        print(f"Created thread: {created['title']} (id {created['id']})")

    posts = [
        ("Best Python tips for beginners", "Usman", "Use list comprehensions!", 3),
        ("Best Python tips for beginners", "Bilal", "f-strings beat .format().", 1),
        ("Flask vs Django in 2026", "Chen", "Flask for small APIs.", 2),
    ]
    for title, author, content, votes in posts:
        post = client.post(
            f"/threads/{threads[title]}/posts",
            json={"author": author, "content": content},
        ).get_json()
        for _ in range(votes):
            post = client.post(f"/posts/{post['id']}/upvote").get_json()
        print(f"Post #{post['id']} by {author}: {post['upvotes']} upvotes — "
              f"{content!r}")

    found = client.get("/threads", query_string={"q": "flask"}).get_json()
    print(f"Search 'flask' found {len(found)} thread(s): "
          f"{[t['title'] for t in found]}")

    thread_id = threads["Flask vs Django in 2026"]
    closed = client.post(f"/threads/{thread_id}/close").get_json()
    print(f"Thread {closed['id']} is now {closed['status']}")
    rejected = client.post(
        f"/threads/{thread_id}/posts",
        json={"author": "Usman", "content": "Too late to the party."},
    )
    print(f"Post to closed thread rejected: {rejected.status_code} "
          f"{rejected.get_json()}")


if __name__ == "__main__":
    main()
