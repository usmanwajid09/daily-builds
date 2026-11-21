"""Demo: build a kanban board, add cards, move one, hit a WIP limit."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kanban_api.app import create_app


def main():
    client = create_app().test_client()

    board = client.post("/boards", json={"name": "Website Redesign"}).get_json()
    print(f"Created board #{board['id']}: {board['name']!r}")

    todo = client.post(
        f"/boards/{board['id']}/columns", json={"name": "To Do"}
    ).get_json()
    doing = client.post(
        f"/boards/{board['id']}/columns",
        json={"name": "Doing", "wip_limit": 2},
    ).get_json()
    done = client.post(
        f"/boards/{board['id']}/columns", json={"name": "Done"}
    ).get_json()
    print(f"Columns: {todo['name']} (#{todo['id']}), "
          f"{doing['name']} (#{doing['id']}, WIP {doing['wip_limit']}), "
          f"{done['name']} (#{done['id']})")

    cards = []
    for title, priority in (("Design homepage", "high"),
                            ("Write copy", "medium"),
                            ("Pick fonts", "low")):
        card = client.post(
            f"/boards/{board['id']}/columns/{todo['id']}/cards",
            json={"title": title, "priority": priority},
        ).get_json()
        cards.append(card)
        print(f"Added card #{card['id']}: {card['title']!r} "
              f"({card['priority']}, position {card['position']})")

    moved = client.post(
        f"/boards/{board['id']}/cards/{cards[0]['id']}/move",
        json={"column_id": doing["id"]},
    ).get_json()
    print(f"Moved {moved['title']!r} to Doing: column "
          f"{moved['column_id']}, position {moved['position']}")
    moved2 = client.post(
        f"/boards/{board['id']}/cards/{cards[1]['id']}/move",
        json={"column_id": doing["id"]},
    ).get_json()
    print(f"Moved {moved2['title']!r} to Doing (WIP now 2/2)")

    blocked = client.post(
        f"/boards/{board['id']}/cards/{cards[2]['id']}/move",
        json={"column_id": doing["id"]},
    )
    print(f"Third card into Doing rejected (WIP limit): "
          f"{blocked.status_code} {blocked.get_json()}")

    finished = client.post(
        f"/boards/{board['id']}/cards/{cards[0]['id']}/move",
        json={"column_id": done["id"], "position": 0},
    ).get_json()
    print(f"Moved {finished['title']!r} to Done at position "
          f"{finished['position']}")

    stats = client.get(f"/boards/{board['id']}/stats").get_json()
    print(f"Stats: {stats['total_cards']} cards, by priority "
          f"{stats['by_priority']}")
    for col in stats["columns"]:
        print(f"  {col['name']}: {col['cards']} card(s), "
              f"WIP {col['wip_limit']}, used {col['wip_used_pct']}%")


if __name__ == "__main__":
    main()
