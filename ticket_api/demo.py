"""Demo: create tickets, walk the status workflow, filter and assign."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ticket_api.app import create_app


def main():
    client = create_app().test_client()

    bug = client.post(
        "/tickets",
        json={
            "title": "Login page crashes on Safari",
            "description": "500 error when submitting the login form.",
            "priority": "high",
        },
    ).get_json()
    print(f"Created ticket #{bug['id']}: {bug['title']!r} "
          f"(priority {bug['priority']}, status {bug['status']})")

    feature = client.post(
        "/tickets",
        json={"title": "Add dark mode", "priority": "low", "assignee": "usman"},
    ).get_json()
    print(f"Created ticket #{feature['id']}: {feature['title']!r} "
          f"(assignee {feature['assignee']})")

    assigned = client.put(
        f"/tickets/{bug['id']}", json={"assignee": "ali"}
    ).get_json()
    print(f"Ticket #{bug['id']} assigned to {assigned['assignee']}")

    for step in ("in_progress", "resolved", "closed"):
        moved = client.post(
            f"/tickets/{bug['id']}/status", json={"status": step}
        ).get_json()
        print(f"Ticket #{bug['id']} -> {moved['status']}")

    reopened = client.post(
        f"/tickets/{bug['id']}/status", json={"status": "in_progress"}
    )
    print(f"Reopening a closed ticket rejected: {reopened.status_code} "
          f"{reopened.get_json()}")

    bad_jump = client.post(
        f"/tickets/{feature['id']}/status", json={"status": "resolved"}
    )
    print(f"Skipping open -> resolved rejected: {bad_jump.status_code} "
          f"{bad_jump.get_json()}")

    open_tickets = client.get("/tickets", query_string={"status": "open"}).get_json()
    print(f"Open tickets: {len(open_tickets)} "
          f"({[t['title'] for t in open_tickets]})")
    mine = client.get("/tickets", query_string={"assignee": "usman"}).get_json()
    print(f"Tickets assigned to usman: {len(mine)}")


if __name__ == "__main__":
    main()
