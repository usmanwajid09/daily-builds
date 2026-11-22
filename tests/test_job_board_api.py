"""Tests for the job_board_api Flask app, using Flask's test client."""
import pytest

from job_board_api.app import create_app

JOB = {
    "title": "Associate Software Engineer",
    "company": "NovaStack",
    "location": "Lahore",
    "job_type": "full-time",
    "salary_min": 70000,
    "salary_max": 120000,
}


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def job(client):
    return client.post("/jobs", json=JOB).get_json()


def _apply(client, job_id, email="usman@example.com"):
    return client.post(
        f"/jobs/{job_id}/applications",
        json={"applicant_name": "Usman", "applicant_email": email},
    )


def test_create_job_returns_201_open(client):
    resp = client.post("/jobs", json=JOB)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["title"] == "Associate Software Engineer"
    assert body["status"] == "open"
    assert body["id"] == 1


def test_create_job_rejects_missing_title(client):
    job = dict(JOB)
    del job["title"]
    assert client.post("/jobs", json=job).status_code == 400


def test_create_job_rejects_bad_job_type(client):
    assert client.post("/jobs", json=dict(JOB, job_type="volunteer")).status_code == 400


def test_create_job_rejects_inverted_salary(client):
    assert client.post(
        "/jobs", json=dict(JOB, salary_min=100000, salary_max=50000)
    ).status_code == 400


def test_create_job_rejects_negative_salary(client):
    assert client.post("/jobs", json=dict(JOB, salary_min=-100)).status_code == 400


def test_get_job_and_404(client, job):
    assert client.get(f"/jobs/{job['id']}").get_json()["company"] == "NovaStack"
    assert client.get("/jobs/999").status_code == 404


def test_list_jobs_filters(client, job):
    client.post(
        "/jobs",
        json={
            "title": "AI Engineer",
            "company": "TechCorp",
            "location": "Remote",
            "job_type": "remote",
        },
    )
    lahore = client.get("/jobs", query_string={"location": "Lahore"}).get_json()
    assert [j["title"] for j in lahore] == ["Associate Software Engineer"]
    remote = client.get("/jobs", query_string={"type": "remote"}).get_json()
    assert [j["title"] for j in remote] == ["AI Engineer"]
    by_company = client.get("/jobs", query_string={"company": "novastack"}).get_json()
    assert len(by_company) == 1


def test_list_jobs_rejects_bad_type_filter(client):
    assert client.get("/jobs", query_string={"type": "volunteer"}).status_code == 400


def test_update_job(client, job):
    resp = client.put(f"/jobs/{job['id']}", json={"location": "Islamabad"})
    assert resp.status_code == 200
    assert resp.get_json()["location"] == "Islamabad"


def test_update_job_rejects_inverted_salary(client, job):
    assert client.put(
        f"/jobs/{job['id']}", json={"salary_max": 1000}
    ).status_code == 400


def test_delete_job_removes_its_applications(client, job):
    _apply(client, job["id"])
    assert client.delete(f"/jobs/{job['id']}").status_code == 200
    assert client.get(f"/jobs/{job['id']}").status_code == 404
    assert client.get("/jobs/1/applications").status_code == 404


def test_close_and_reopen(client, job):
    closed = client.post(f"/jobs/{job['id']}/close")
    assert closed.status_code == 200
    assert closed.get_json()["status"] == "closed"
    assert client.post(f"/jobs/{job['id']}/close").status_code == 409
    reopened = client.post(f"/jobs/{job['id']}/reopen")
    assert reopened.get_json()["status"] == "open"
    assert client.post(f"/jobs/{job['id']}/reopen").status_code == 409


def test_apply_returns_201_submitted(client, job):
    resp = _apply(client, job["id"])
    assert resp.status_code == 201
    assert resp.get_json()["status"] == "submitted"


def test_apply_to_closed_job_returns_409(client, job):
    client.post(f"/jobs/{job['id']}/close")
    assert _apply(client, job["id"]).status_code == 409


def test_apply_requires_valid_email(client, job):
    resp = client.post(
        f"/jobs/{job['id']}/applications",
        json={"applicant_name": "Usman", "applicant_email": "bad-email"},
    )
    assert resp.status_code == 400


def test_duplicate_application_returns_409(client, job):
    assert _apply(client, job["id"]).status_code == 201
    assert _apply(client, job["id"]).status_code == 409
    # A different email may still apply.
    assert _apply(client, job["id"], "other@example.com").status_code == 201


def test_application_status_workflow(client, job):
    application = _apply(client, job["id"]).get_json()
    for step in ("reviewed", "shortlisted", "hired"):
        resp = client.post(
            f"/applications/{application['id']}/status", json={"status": step}
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == step


def test_application_cannot_skip_steps(client, job):
    application = _apply(client, job["id"]).get_json()
    resp = client.post(
        f"/applications/{application['id']}/status", json={"status": "hired"}
    )
    assert resp.status_code == 409


def test_application_reject_is_terminal(client, job):
    application = _apply(client, job["id"]).get_json()
    assert client.post(
        f"/applications/{application['id']}/status", json={"status": "rejected"}
    ).status_code == 200
    assert client.post(
        f"/applications/{application['id']}/status", json={"status": "reviewed"}
    ).status_code == 409


def test_application_bad_status_value_400(client, job):
    application = _apply(client, job["id"]).get_json()
    assert client.post(
        f"/applications/{application['id']}/status", json={"status": "maybe"}
    ).status_code == 400


def test_application_404(client):
    assert client.get("/applications/999").status_code == 404
    assert client.post(
        "/applications/999/status", json={"status": "reviewed"}
    ).status_code == 404


def test_list_job_applications(client, job):
    _apply(client, job["id"], "a@example.com")
    _apply(client, job["id"], "b@example.com")
    applications = client.get(f"/jobs/{job['id']}/applications").get_json()
    assert len(applications) == 2
    assert client.get("/jobs/999/applications").status_code == 404
