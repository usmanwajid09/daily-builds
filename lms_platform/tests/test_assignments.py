from lms_platform.tests.conftest import auth_header, signup


def _instructor(client, email="prof@example.com"):
    return signup(client, email, "password123", "instructor")


def _student(client, email="stud@example.com"):
    return signup(client, email, "password123", "student")


def _make_course(client, token):
    r = client.post("/api/courses", json={"title": "C1"}, headers=auth_header(token))
    return r.get_json()["id"]


def test_instructor_can_create_assignment(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    r = client.post(f"/api/courses/{course_id}/assignments",
                     json={"title": "HW1", "description": "do it", "max_points": 20},
                     headers=auth_header(token))
    assert r.status_code == 201
    body = r.get_json()
    assert body["title"] == "HW1"
    assert body["max_points"] == 20


def test_student_cannot_create_assignment(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    stoken, _ = _student(client)
    r = client.post(f"/api/courses/{course_id}/assignments", json={"title": "HW1"},
                     headers=auth_header(stoken))
    assert r.status_code == 403


def test_assignment_requires_title(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    r = client.post(f"/api/courses/{course_id}/assignments", json={}, headers=auth_header(token))
    assert r.status_code == 400


def test_assignment_max_points_must_be_positive(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    r = client.post(f"/api/courses/{course_id}/assignments",
                     json={"title": "HW1", "max_points": 0}, headers=auth_header(token))
    assert r.status_code == 400


def test_submit_requires_enrollment(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    r = client.post(f"/api/courses/{course_id}/assignments", json={"title": "HW1"},
                     headers=auth_header(itoken))
    assignment_id = r.get_json()["id"]
    stoken, _ = _student(client)
    r = client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "x"},
                     headers=auth_header(stoken))
    assert r.status_code == 403


def _enrolled_assignment(client, itoken, stoken, course_id, max_points=100):
    r = client.post(f"/api/courses/{course_id}/assignments",
                     json={"title": "HW1", "max_points": max_points},
                     headers=auth_header(itoken))
    assignment_id = r.get_json()["id"]
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))
    return assignment_id


def test_submit_then_resubmit_overwrites_content(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    assignment_id = _enrolled_assignment(client, itoken, stoken, course_id)

    r = client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "v1"},
                     headers=auth_header(stoken))
    assert r.status_code == 200
    assert r.get_json()["content"] == "v1"

    r = client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "v2"},
                     headers=auth_header(stoken))
    assert r.get_json()["content"] == "v2"

    r = client.get(f"/api/assignments/{assignment_id}/submissions", headers=auth_header(itoken))
    assert len(r.get_json()["submissions"]) == 1  # overwritten, not duplicated


def test_resubmission_clears_prior_grade(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, sid = _student(client)
    assignment_id = _enrolled_assignment(client, itoken, stoken, course_id)

    client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "v1"},
                headers=auth_header(stoken))
    r = client.patch(f"/api/assignments/{assignment_id}/submissions/{sid}/grade",
                      json={"grade": 95, "feedback": "great"}, headers=auth_header(itoken))
    assert r.get_json()["grade"] == 95

    r = client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "v2"},
                     headers=auth_header(stoken))
    assert r.get_json()["grade"] is None
    assert r.get_json()["feedback"] is None


def test_grade_must_be_within_max_points(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, sid = _student(client)
    assignment_id = _enrolled_assignment(client, itoken, stoken, course_id, max_points=10)
    client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "v1"},
                headers=auth_header(stoken))

    r = client.patch(f"/api/assignments/{assignment_id}/submissions/{sid}/grade",
                      json={"grade": 11}, headers=auth_header(itoken))
    assert r.status_code == 400

    r = client.patch(f"/api/assignments/{assignment_id}/submissions/{sid}/grade",
                      json={"grade": -1}, headers=auth_header(itoken))
    assert r.status_code == 400

    r = client.patch(f"/api/assignments/{assignment_id}/submissions/{sid}/grade",
                      json={"grade": 10}, headers=auth_header(itoken))
    assert r.status_code == 200


def test_grade_requires_existing_submission(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    r = client.post(f"/api/courses/{course_id}/assignments", json={"title": "HW1"},
                     headers=auth_header(itoken))
    assignment_id = r.get_json()["id"]
    r = client.patch(f"/api/assignments/{assignment_id}/submissions/999/grade",
                      json={"grade": 5}, headers=auth_header(itoken))
    assert r.status_code == 404


def test_non_owner_instructor_cannot_grade(client):
    itoken, _ = _instructor(client, "a@example.com")
    course_id = _make_course(client, itoken)
    stoken, sid = _student(client)
    assignment_id = _enrolled_assignment(client, itoken, stoken, course_id)
    client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "v1"},
                headers=auth_header(stoken))

    other, _ = _instructor(client, "b@example.com")
    r = client.patch(f"/api/assignments/{assignment_id}/submissions/{sid}/grade",
                      json={"grade": 5}, headers=auth_header(other))
    assert r.status_code == 404


def test_me_submissions_requires_assignment_id(client):
    stoken, _ = _student(client)
    r = client.get("/api/me/submissions", headers=auth_header(stoken))
    assert r.status_code == 400


def test_me_submissions_returns_own_submission_for_assignment(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    assignment_id = _enrolled_assignment(client, itoken, stoken, course_id)
    client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "v1"},
                headers=auth_header(stoken))
    r = client.get(f"/api/me/submissions?assignment_id={assignment_id}",
                    headers=auth_header(stoken))
    assert r.status_code == 200
    subs = r.get_json()["submissions"]
    assert len(subs) == 1
    assert subs[0]["content"] == "v1"
