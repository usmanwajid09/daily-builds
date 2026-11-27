"""Demo: create a project, add tasks, move them through the workflow."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_manager_api.app import create_app


def main():
    client = create_app().test_client()

    project = client.post(
        "/projects",
        json={"name": "Website Redesign", "description": "Rebuild the company site"},
    ).get_json()
    print(f"Created project #{project['id']}: {project['name']!r}")

    tasks = []
    for title, assignee, due in (
        ("Design homepage", "Ayesha", "2026-12-15"),
        ("Write copy", "Bilal", "2026-12-20"),
        ("Deploy to staging", "Ayesha", "2020-01-01"),
    ):
        task = client.post(
            f"/projects/{project['id']}/tasks",
            json={"title": title, "assignee": assignee, "due_date": due},
        ).get_json()
        tasks.append(task)
        print(f"Added task #{task['id']}: {task['title']!r} "
              f"({task['status']}, due {task['due_date']}, "
              f"overdue={task['overdue']})")

    moved = client.post(
        f"/projects/{project['id']}/tasks/{tasks[0]['id']}/status",
        json={"status": "in_progress"},
    ).get_json()
    print(f"Moved {moved['title']!r} to {moved['status']}")
    done = client.post(
        f"/projects/{project['id']}/tasks/{tasks[0]['id']}/status",
        json={"status": "done"},
    ).get_json()
    print(f"Moved {done['title']!r} to {done['status']}")

    reopened = client.post(
        f"/projects/{project['id']}/tasks/{tasks[0]['id']}/status",
        json={"status": "in_progress"},
    )
    print(f"done -> in_progress rejected (workflow): "
          f"{reopened.status_code} {reopened.get_json()}")

    progress = client.get(f"/projects/{project['id']}/progress").get_json()
    print(f"Progress: {progress['done_tasks']}/{progress['total_tasks']} done "
          f"({progress['progress_pct']}%), "
          f"{progress['overdue_count']} overdue "
          f"(ids {progress['overdue_task_ids']}), "
          f"by status {progress['by_status']}")


if __name__ == "__main__":
    main()
