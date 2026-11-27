"""Tests for the job_tracker_api Flask app, using Flask's test client."""
import pytest

from job_tracker_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _application(client, company="Acme Corp", role="Software Engineer", **extra):
    payload = {"company": company, "role": role}
    payload.update(extra)
    return client.post("/applications", json=payload)


@pytest.fixture
def application(client):
    return _application(client).get_json()["id"]


def test_list_applications_starts_empty(client):
    assert client.get("/applications").get_json() == []


def test_create_application_returns_201(client):
    resp = _application(
        client, salary=90000, applied_date="2026-11-20", notes="Via referral"
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["status"] == "applied"
    assert body["salary"] == 90000
    assert body["interview_date"] is None


def test_create_application_validation_400(client):
    assert _application(client, company="").status_code == 400
    assert _application(client, role=42).status_code == 400
    assert _application(client, status="hired").status_code == 400
    assert _application(client, applied_date="20-11-2026").status_code == 400
    assert _application(client, interview_date="2026-02-30").status_code == 400
    assert _application(client, salary=-1).status_code == 400
    assert _application(client, notes=7).status_code == 400


def test_get_application_404(client):
    assert client.get("/applications/99").status_code == 404


def test_update_application(client, application):
    resp = client.put(
        f"/applications/{application}",
        json={"salary": 100000, "interview_date": "2026-12-05"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["salary"] == 100000
    assert body["interview_date"] == "2026-12-05"
    assert client.put(f"/applications/{application}", json={}).status_code == 400
    assert client.put(
        f"/applications/{application}", json={"status": "offer"}
    ).status_code == 400
    assert client.put(
        f"/applications/{application}", json={"salary": -5}
    ).status_code == 400


def test_delete_application(client, application):
    assert client.delete(f"/applications/{application}").status_code == 200
    assert client.get(f"/applications/{application}").status_code == 404


def test_status_pipeline_happy_path(client, application):
    url = f"/applications/{application}/status"
    assert client.post(url, json={"status": "screening"}).status_code == 200
    assert client.post(url, json={"status": "interview"}).status_code == 200
    resp = client.post(url, json={"status": "offer"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "offer"


def test_status_pipeline_invalid_moves_409(client, application):
    url = f"/applications/{application}/status"
    assert client.post(url, json={"status": "interview"}).status_code == 200
    # Backwards: interview -> applied is not allowed.
    assert client.post(url, json={"status": "applied"}).status_code == 409
    assert client.post(url, json={"status": "interview"}).status_code == 409
    assert client.post(url, json={"status": "hired"}).status_code == 400


def test_status_rejected_is_terminal(client, application):
    url = f"/applications/{application}/status"
    assert client.post(url, json={"status": "rejected"}).status_code == 200
    assert client.post(url, json={"status": "interview"}).status_code == 409
    assert client.post(url, json={"status": "applied"}).status_code == 409


def test_status_offer_can_be_declined(client, application):
    url = f"/applications/{application}/status"
    assert client.post(url, json={"status": "offer"}).status_code == 200
    assert client.post(url, json={"status": "rejected"}).status_code == 200


def test_list_filters(client):
    _application(client, company="Acme Corp", status="screening")
    _application(client, company="Globex", status="offer")
    _application(client, company="Acme Labs", status="applied")
    assert len(client.get("/applications").get_json()) == 3
    assert len(client.get("/applications?status=offer").get_json()) == 1
    assert len(client.get("/applications?company=acme").get_json()) == 2
    assert client.get("/applications?status=nope").status_code == 400


def test_stats(client):
    _application(client, company="A", status="applied")
    _application(client, company="B", status="interview")
    _application(client, company="C", status="offer")
    _application(client, company="D", status="rejected")
    stats = client.get("/stats").get_json()
    assert stats["total_applications"] == 4
    assert stats["by_status"] == {
        "applied": 1, "screening": 0, "interview": 1,
        "offer": 1, "rejected": 1,
    }
    assert stats["offer_rate"] == 25.0
    assert stats["interview_rate"] == 25.0


def test_stats_empty(client):
    stats = client.get("/stats").get_json()
    assert stats["total_applications"] == 0
    assert stats["offer_rate"] == 0.0
    assert stats["interview_rate"] == 0.0


def test_upcoming_interviews(client):
    past = _application(
        client, company="Past Co", status="interview",
        interview_date="2020-01-01",
    ).get_json()
    soon = _application(
        client, company="Soon Co", status="interview",
        interview_date="2999-01-10",
    ).get_json()
    sooner = _application(
        client, company="Sooner Co", status="interview",
        interview_date="2999-01-05",
    ).get_json()
    _application(
        client, company="No Date Co", status="interview",
    )
    _application(
        client, company="Offer Co", status="offer",
        interview_date="2999-01-01",
    )
    upcoming = client.get("/interviews/upcoming").get_json()
    # Sorted by date; past interviews and interviews without a date
    # (or not in interview status) are excluded.
    assert [a["id"] for a in upcoming] == [sooner["id"], soon["id"]]
    assert past["id"] not in [a["id"] for a in upcoming]
