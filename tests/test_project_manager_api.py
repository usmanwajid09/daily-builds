"""Tests for the project_manager_api Flask app, using Flask's test client."""
import pytest

from project_manager_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _project(client, name="Website Redesign", **extra):
    payload = {"name": name}
    payload.update(extra)
    return client.post("/projects", json=payload)


def _task(client, project_id, title="Do thing", **extra):
    payload = {"title": title}
    payload.update(extra)
    return client.post(f"/projects/{project_id}/tasks", json=payload)


@pytest.fixture
def project(client):
    return _project(client).get_json()["id"]


def test_list_projects_starts_empty(client):
    assert client.get("/projects").get_json() == []


def test_create_project_returns_201(client):
    resp = _project(client, description="Rebuild the site")
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["name"] == "Website Redesign"
    assert body["total_tasks"] == 0
    assert body["progress_pct"] == 0.0


def test_create_project_missing_name_400(client):
    assert client.post("/projects", json={}).status_code == 400
    assert client.post("/projects", json={"name": "  "}).status_code == 400
    assert client.post("/projects", json={"name": 42}).status_code == 400


def test_create_project_duplicate_name_409(client, project):
    assert _project(client, name="website redesign").status_code == 409


def test_get_project_404(client):
    assert client.get("/projects/99").status_code == 404


def test_update_project(client, project):
    resp = client.put(f"/projects/{project}", json={"name": "New Name"})
    assert resp.status_code == 200
    assert resp.get_json()["name"] == "New Name"


def test_update_project_duplicate_name_409(client, project):
    _project(client, name="Other")
    assert client.put(
        f"/projects/{project}", json={"name": "other"}
    ).status_code == 409


def test_delete_project(client, project):
    assert client.delete(f"/projects/{project}").status_code == 200
    assert client.get(f"/projects/{project}").status_code == 404


def test_create_task_returns_201(client, project):
    resp = _task(client, project, assignee="Ayesha", due_date="2026-12-15")
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["status"] == "todo"
    assert body["assignee"] == "Ayesha"
    assert body["overdue"] is False


def test_create_task_validation_400(client, project):
    assert _task(client, project, title="").status_code == 400
    assert _task(client, project, due_date="15-12-2026").status_code == 400
    assert _task(client, project, due_date="2026-13-01").status_code == 400
    assert _task(client, project, status="blocked").status_code == 400
    assert client.post(
        f"/projects/{project}/tasks", json={"title": "x", "assignee": 7}
    ).status_code == 400


def test_create_task_unknown_project_404(client):
    assert _task(client, 99).status_code == 404


def test_task_overdue_flag(client, project):
    past = _task(client, project, due_date="2020-01-01").get_json()
    assert past["overdue"] is True
    done = client.post(
        f"/projects/{project}/tasks/{past['id']}/status",
        json={"status": "done"},
    ).get_json()
    assert done["overdue"] is False
    future = _task(client, project, due_date="2999-01-01").get_json()
    assert future["overdue"] is False


def test_list_tasks_status_filter(client, project):
    _task(client, project, title="A")
    done_task = _task(client, project, title="B").get_json()
    client.post(
        f"/projects/{project}/tasks/{done_task['id']}/status",
        json={"status": "done"},
    )
    assert len(client.get(f"/projects/{project}/tasks").get_json()) == 2
    done_only = client.get(
        f"/projects/{project}/tasks?status=done"
    ).get_json()
    assert [t["id"] for t in done_only] == [done_task["id"]]
    assert client.get(
        f"/projects/{project}/tasks?status=nope"
    ).status_code == 400


def test_get_task_404(client, project):
    assert client.get(f"/projects/{project}/tasks/99").status_code == 404


def test_update_task(client, project):
    task = _task(client, project).get_json()
    resp = client.put(
        f"/projects/{project}/tasks/{task['id']}",
        json={"title": "Renamed", "assignee": "Bilal"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["title"] == "Renamed"
    assert resp.get_json()["assignee"] == "Bilal"
    assert client.put(
        f"/projects/{project}/tasks/{task['id']}", json={}
    ).status_code == 400


def test_status_workflow_happy_path(client, project):
    task = _task(client, project).get_json()
    url = f"/projects/{project}/tasks/{task['id']}/status"
    assert client.post(url, json={"status": "in_progress"}).status_code == 200
    resp = client.post(url, json={"status": "done"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "done"


def test_status_workflow_invalid_moves_409(client, project):
    task = _task(client, project).get_json()
    url = f"/projects/{project}/tasks/{task['id']}/status"
    assert client.post(url, json={"status": "done"}).status_code == 200
    assert client.post(url, json={"status": "in_progress"}).status_code == 409
    assert client.post(url, json={"status": "done"}).status_code == 409
    assert client.post(url, json={"status": "nope"}).status_code == 400


def test_delete_task(client, project):
    task = _task(client, project).get_json()
    assert client.delete(
        f"/projects/{project}/tasks/{task['id']}"
    ).status_code == 200
    assert client.get(
        f"/projects/{project}/tasks/{task['id']}"
    ).status_code == 404


def test_progress_stats(client, project):
    _task(client, project, title="A", due_date="2020-01-01")
    done_task = _task(client, project, title="B").get_json()
    _task(client, project, title="C")
    client.post(
        f"/projects/{project}/tasks/{done_task['id']}/status",
        json={"status": "done"},
    )
    progress = client.get(f"/projects/{project}/progress").get_json()
    assert progress["total_tasks"] == 3
    assert progress["done_tasks"] == 1
    assert progress["progress_pct"] == 33.3
    assert progress["overdue_count"] == 1
    assert progress["by_status"] == {
        "todo": 2, "in_progress": 0, "done": 1,
    }


def test_project_list_shows_progress(client, project):
    _task(client, project)
    done_task = _task(client, project, title="B").get_json()
    client.post(
        f"/projects/{project}/tasks/{done_task['id']}/status",
        json={"status": "done"},
    )
    listed = client.get("/projects").get_json()
    assert listed[0]["total_tasks"] == 2
    assert listed[0]["done_tasks"] == 1
    assert listed[0]["progress_pct"] == 50.0
