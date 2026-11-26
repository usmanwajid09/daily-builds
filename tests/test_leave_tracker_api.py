"""Tests for the leave_tracker_api Flask app, using Flask's test client."""
import pytest

from leave_tracker_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _create(client, name="Sara", balances=None):
    body = {"name": name}
    if balances is not None:
        body["balances"] = balances
    return client.post("/employees", json=body)


def _request(client, employee_id, leave_type="annual", start="2026-12-07", end="2026-12-11"):
    return client.post(
        f"/employees/{employee_id}/leave",
        json={"type": leave_type, "start_date": start, "end_date": end},
    )


def test_list_starts_empty(client):
    assert client.get("/employees").get_json() == []


def test_create_employee_default_balances(client):
    resp = _create(client, "Sara")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "name": "Sara",
        "balances": {"annual": 20, "sick": 10, "casual": 7},
    }


def test_create_employee_custom_balances(client):
    resp = _create(client, "Omar", {"annual": 25, "sick": 5})
    assert resp.status_code == 201
    assert resp.get_json()["balances"] == {"annual": 25, "sick": 5, "casual": 7}


def test_create_requires_non_empty_name(client):
    assert _create(client, "").status_code == 400
    assert _create(client, "   ").status_code == 400
    assert client.post("/employees", json={}).status_code == 400


def test_create_rejects_bad_balances(client):
    assert _create(client, "Sara", {"annual": -1}).status_code == 400
    assert _create(client, "Sara", {"annual": 1.5}).status_code == 400
    assert _create(client, "Sara", {"holiday": 5}).status_code == 400
    assert _create(client, "Sara", {"annual": "20"}).status_code == 400


def test_get_employee_and_404(client):
    employee_id = _create(client).get_json()["id"]
    assert client.get(f"/employees/{employee_id}").get_json()["name"] == "Sara"
    assert client.get("/employees/999").status_code == 404


def test_request_leave_counts_days_inclusively(client):
    employee_id = _create(client).get_json()["id"]
    resp = _request(client, employee_id)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["days"] == 5
    assert body["status"] == "pending"
    assert body["start_date"] == "2026-12-07"
    assert body["end_date"] == "2026-12-11"


def test_approve_deducts_balance(client):
    employee_id = _create(client).get_json()["id"]
    request_id = _request(client, employee_id).get_json()["id"]
    approved = client.post(f"/leave/{request_id}/approve")
    assert approved.status_code == 200
    assert approved.get_json()["status"] == "approved"
    balance = client.get(f"/employees/{employee_id}/balance").get_json()
    assert balance["balances"]["annual"] == 15
    assert balance["balances"]["sick"] == 10


def test_reject_leaves_balance_untouched(client):
    employee_id = _create(client).get_json()["id"]
    request_id = _request(client, employee_id).get_json()["id"]
    rejected = client.post(f"/leave/{request_id}/reject")
    assert rejected.get_json()["status"] == "rejected"
    balance = client.get(f"/employees/{employee_id}/balance").get_json()
    assert balance["balances"]["annual"] == 20


def test_deciding_twice_returns_409(client):
    employee_id = _create(client).get_json()["id"]
    request_id = _request(client, employee_id).get_json()["id"]
    client.post(f"/leave/{request_id}/approve")
    again = client.post(f"/leave/{request_id}/approve")
    assert again.status_code == 409
    assert "already" in again.get_json()["error"]
    assert client.post(f"/leave/{request_id}/reject").status_code == 409


def test_overlapping_request_returns_409(client):
    employee_id = _create(client).get_json()["id"]
    _request(client, employee_id)
    overlap = _request(client, employee_id, start="2026-12-10", end="2026-12-12")
    assert overlap.status_code == 409
    assert "overlap" in overlap.get_json()["error"]
    # A rejected request frees the dates up again.
    request_id = _request(client, employee_id, start="2026-12-14", end="2026-12-15").get_json()["id"]
    client.post(f"/leave/{request_id}/reject")
    retry = _request(client, employee_id, start="2026-12-14", end="2026-12-15")
    assert retry.status_code == 201


def test_insufficient_balance_returns_400(client):
    employee_id = _create(client).get_json()["id"]
    resp = _request(client, employee_id, start="2027-01-05", end="2027-01-25")
    assert resp.status_code == 400
    assert "insufficient" in resp.get_json()["error"]


def test_invalid_type_and_dates_return_400(client):
    employee_id = _create(client).get_json()["id"]
    assert _request(client, employee_id, leave_type="holiday").status_code == 400
    assert _request(client, employee_id, start="bad-date").status_code == 400
    assert _request(client, employee_id, start="2026-12-11", end="2026-12-07").status_code == 400
    assert _request(client, employee_id, start="2026-02-30", end="2026-03-01").status_code == 400


def test_request_on_unknown_employee_is_404(client):
    assert _request(client, 999).status_code == 404
    assert client.get("/employees/999/balance").status_code == 404


def test_list_leave_requests_and_get_one(client):
    employee_id = _create(client).get_json()["id"]
    first = _request(client, employee_id).get_json()["id"]
    second = _request(client, employee_id, leave_type="sick", start="2026-12-14", end="2026-12-14").get_json()["id"]
    listed = client.get(f"/employees/{employee_id}/leave").get_json()
    assert [r["id"] for r in listed] == [first, second]
    assert client.get(f"/leave/{first}").get_json()["type"] == "annual"
    assert client.get("/leave/999").status_code == 404


def test_decide_unknown_request_is_404(client):
    assert client.post("/leave/999/approve").status_code == 404
    assert client.post("/leave/999/reject").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
