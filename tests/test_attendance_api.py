"""Tests for the attendance_api Flask app, using Flask's test client."""
import pytest

from attendance_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _create(client, name="Ayesha"):
    return client.post("/students", json={"name": name})


def _mark(client, student_id, day, status):
    return client.post(
        f"/students/{student_id}/marks", json={"date": day, "status": status}
    )


def test_list_starts_empty(client):
    assert client.get("/students").get_json() == []


def test_create_student_returns_201(client):
    resp = _create(client, "Ayesha")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "name": "Ayesha",
        "marks": {},
        "total_marks": 0,
        "attendance_percentage": 0.0,
    }


def test_create_requires_non_empty_name(client):
    assert _create(client, "").status_code == 400
    assert _create(client, "   ").status_code == 400
    assert client.post("/students", json={}).status_code == 400
    assert client.post("/students", json={"name": 42}).status_code == 400


def test_get_student_and_404(client):
    student_id = _create(client).get_json()["id"]
    assert client.get(f"/students/{student_id}").get_json()["name"] == "Ayesha"
    assert client.get("/students/999").status_code == 404


def test_mark_present_and_absent(client):
    student_id = _create(client).get_json()["id"]
    assert _mark(client, student_id, "2026-11-23", "present").status_code == 201
    resp = _mark(client, student_id, "2026-11-24", "absent")
    assert resp.status_code == 201
    assert resp.get_json()["marks"] == {
        "2026-11-23": "present",
        "2026-11-24": "absent",
    }


def test_attendance_percentage_counts_late_as_attended(client):
    student_id = _create(client).get_json()["id"]
    _mark(client, student_id, "2026-11-23", "present")
    _mark(client, student_id, "2026-11-24", "late")
    _mark(client, student_id, "2026-11-25", "absent")
    report = client.get(f"/students/{student_id}/attendance").get_json()
    assert report == {
        "student_id": student_id,
        "name": "Ayesha",
        "total_days": 3,
        "present": 1,
        "absent": 1,
        "late": 1,
        "attendance_percentage": 66.67,
    }


def test_attendance_report_for_unmarked_student(client):
    student_id = _create(client).get_json()["id"]
    report = client.get(f"/students/{student_id}/attendance").get_json()
    assert report["total_days"] == 0
    assert report["attendance_percentage"] == 0.0


def test_attendance_report_404_for_unknown_student(client):
    assert client.get("/students/999/attendance").status_code == 404


def test_duplicate_mark_returns_409(client):
    student_id = _create(client).get_json()["id"]
    _mark(client, student_id, "2026-11-23", "present")
    dup = _mark(client, student_id, "2026-11-23", "absent")
    assert dup.status_code == 409
    assert "already" in dup.get_json()["error"]


def test_invalid_status_returns_400(client):
    student_id = _create(client).get_json()["id"]
    assert _mark(client, student_id, "2026-11-23", "half-day").status_code == 400
    assert _mark(client, student_id, "2026-11-23", "").status_code == 400
    assert _mark(client, student_id, "2026-11-23", None).status_code == 400


def test_invalid_date_returns_400(client):
    student_id = _create(client).get_json()["id"]
    assert _mark(client, student_id, "not-a-date", "present").status_code == 400
    assert _mark(client, student_id, "2026-02-30", "present").status_code == 400
    assert _mark(client, student_id, "23-11-2026", "present").status_code == 400


def test_mark_on_unknown_student_is_404(client):
    assert _mark(client, 999, "2026-11-23", "present").status_code == 404


def test_summary_aggregates_class(client):
    a = _create(client, "Ayesha").get_json()["id"]
    b = _create(client, "Bilal").get_json()["id"]
    _mark(client, a, "2026-11-23", "present")
    _mark(client, a, "2026-11-24", "absent")
    _mark(client, b, "2026-11-23", "present")
    summary = client.get("/summary").get_json()
    assert summary["total_students"] == 2
    assert summary["total_marks"] == 3
    assert summary["overall_attendance_percentage"] == 66.67
    assert [s["attendance_percentage"] for s in summary["students"]] == [50.0, 100.0]


def test_summary_with_no_students(client):
    summary = client.get("/summary").get_json()
    assert summary == {
        "total_students": 0,
        "total_marks": 0,
        "overall_attendance_percentage": 0.0,
        "students": [],
    }


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
