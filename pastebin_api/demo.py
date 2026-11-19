"""Demo: create snippets, one expiring, and show the 410 on expiry."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pastebin_api.app import create_app


def main():
    client = create_app().test_client()

    first = client.post(
        "/snippets", json={"content": "print('hello world')"}
    ).get_json()
    print(f"Created snippet {first['id']}: expires_at={first['expires_at']}")
    short = client.post(
        "/snippets", json={"content": "temporary note", "expires_in": 1}
    ).get_json()
    print(f"Created short-lived snippet {short['id']}: expires in 1s")

    live = client.get("/snippets").get_json()
    print(f"{len(live)} live snippet(s): {[s['id'] for s in live]}")

    time.sleep(1.2)
    gone = client.get(f"/snippets/{short['id']}")
    print(f"Expired snippet fetch: {gone.status_code} {gone.get_json()}")
    live = client.get("/snippets").get_json()
    print(f"After expiry, {len(live)} live snippet(s): {[s['id'] for s in live]}")


if __name__ == "__main__":
    main()
