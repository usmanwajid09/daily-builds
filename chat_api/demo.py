"""Demo: create rooms, post messages, read per-room history."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_api.app import create_app


def main():
    client = create_app().test_client()

    general = client.post("/rooms", json={"name": "general"}).get_json()
    random = client.post("/rooms", json={"name": "random"}).get_json()
    print(f"Created rooms: {general['name']} (id {general['id']}), "
          f"{random['name']} (id {random['id']})")

    client.post(f"/rooms/{general['id']}/messages",
                json={"sender": "usman", "text": "Hello everyone!"})
    client.post(f"/rooms/{general['id']}/messages",
                json={"sender": "ali", "text": "Hi Usman, welcome."})
    client.post(f"/rooms/{general['id']}/messages",
                json={"sender": "usman", "text": "Working on the chat API today."})
    client.post(f"/rooms/{random['id']}/messages",
                json={"sender": "usman", "text": "Off-topic goes here."})

    history = client.get(f"/rooms/{general['id']}/messages").get_json()
    print(f"Room 'general' history: {len(history)} message(s)")
    for m in history:
        print(f"  [{m['sender']}] {m['text']}")

    mine = client.get(f"/rooms/{general['id']}/messages",
                      query_string={"sender": "usman"}).get_json()
    print(f"Messages from 'usman' in general: {len(mine)}")

    latest = client.get(f"/rooms/{general['id']}/messages",
                        query_string={"limit": 2}).get_json()
    print(f"First 2 messages: {[m['text'] for m in latest]}")

    bad = client.post(f"/rooms/{general['id']}/messages",
                      json={"sender": "  ", "text": "no sender"})
    print(f"Empty sender rejected: {bad.status_code} {bad.get_json()}")

    room = client.get(f"/rooms/{general['id']}").get_json()
    print(f"Room 'general' now has message_count={room['message_count']}")


if __name__ == "__main__":
    main()
