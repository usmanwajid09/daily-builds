from lms_platform.tests.conftest import auth_header, signup


def _make_instructor(client, email="prof@example.com"):
    return signup(client, email, "password123", "instructor")


def _make_student(client, email="stud@example.com"):
    return signup(client, email, "password123", "student")


def test_instructor_can_create_course(client):
    token, _ = _make_instructor(client)
    r = client.post("/api/courses", json={"title": "Intro", "description": "basics"}, headers=auth_header(token))
    assert r.status_code == 201
    body = r.get_json()
    assert body["title"] == "Intro"
    assert body["description"] == "basics"


def test_student_cannot_create_course(client):
    token, _ = _make_student(client)
    r = client.post("/api/courses", json={"title": "Intro"}, headers=auth_header(token))
    assert r.status_code == 403


def test_create_course_requires_title(client):
    token, _ = _make_instructor(client)
    r = client.post("/api/courses", json={"description": "no title"}, headers=auth_header(token))
    assert r.status_code == 400


def test_list_courses_shows_everyone_the_full_catalogue(client):
    token, _ = _make_instructor(client)
    client.post("/api/courses", json={"title": "Course A"}, headers=auth_header(token))
    stud_token, _ = _make_student(client)
    r = client.get("/api/courses", headers=auth_header(stud_token))
    assert r.status_code == 200
    assert len(r.get_json()["courses"]) == 1


def test_list_mine_filters_to_own_courses(client):
    token1, _ = _make_instructor(client, "p1@example.com")
    token2, _ = _make_instructor(client, "p2@example.com")
    client.post("/api/courses", json={"title": "A"}, headers=auth_header(token1))
    client.post("/api/courses", json={"title": "B"}, headers=auth_header(token2))

    r = client.get("/api/courses?mine=1", headers=auth_header(token1))
    titles = [c["title"] for c in r.get_json()["courses"]]
    assert titles == ["A"]


def test_student_cannot_use_mine_filter(client):
    token, _ = _make_student(client)
    r = client.get("/api/courses?mine=1", headers=auth_header(token))
    assert r.status_code == 403


def test_get_course_not_found(client):
    token, _ = _make_instructor(client)
    r = client.get("/api/courses/999", headers=auth_header(token))
    assert r.status_code == 404


def test_owner_can_update_course(client):
    token, _ = _make_instructor(client)
    course = client.post("/api/courses", json={"title": "Old"}, headers=auth_header(token)).get_json()
    r = client.patch(f"/api/courses/{course['id']}", json={"title": "New"}, headers=auth_header(token))
    assert r.status_code == 200
    assert r.get_json()["title"] == "New"


def test_non_owner_instructor_cannot_update_course(client):
    token1, _ = _make_instructor(client, "p1@example.com")
    token2, _ = _make_instructor(client, "p2@example.com")
    course = client.post("/api/courses", json={"title": "Old"}, headers=auth_header(token1)).get_json()
    r = client.patch(f"/api/courses/{course['id']}", json={"title": "Hijacked"}, headers=auth_header(token2))
    assert r.status_code == 404  # not leaking existence to a non-owner


def test_update_course_rejects_blank_title(client):
    token, _ = _make_instructor(client)
    course = client.post("/api/courses", json={"title": "Old"}, headers=auth_header(token)).get_json()
    r = client.patch(f"/api/courses/{course['id']}", json={"title": "   "}, headers=auth_header(token))
    assert r.status_code == 400


def test_owner_can_delete_course(client):
    token, _ = _make_instructor(client)
    course = client.post("/api/courses", json={"title": "Doomed"}, headers=auth_header(token)).get_json()
    r = client.delete(f"/api/courses/{course['id']}", headers=auth_header(token))
    assert r.status_code == 200
    assert client.get(f"/api/courses/{course['id']}", headers=auth_header(token)).status_code == 404


def test_delete_course_cascades_lessons_and_enrollments(client):
    token, _ = _make_instructor(client)
    course = client.post("/api/courses", json={"title": "Doomed"}, headers=auth_header(token)).get_json()
    lesson = client.post(
        f"/api/courses/{course['id']}/lessons",
        json={"title": "L1", "content_type": "text", "content": "hi"},
        headers=auth_header(token),
    ).get_json()
    stud_token, _ = _make_student(client)
    client.post(f"/api/courses/{course['id']}/enroll", headers=auth_header(stud_token))

    r = client.delete(f"/api/courses/{course['id']}", headers=auth_header(token))
    assert r.status_code == 200

    assert client.get(f"/api/lessons/{lesson['id']}", headers=auth_header(token)).status_code == 404
    r = client.get("/api/me/enrollments", headers=auth_header(stud_token))
    assert r.get_json()["enrollments"] == []


def test_enroll_is_idempotent(client):
    token, _ = _make_instructor(client)
    course = client.post("/api/courses", json={"title": "C"}, headers=auth_header(token)).get_json()
    stud_token, _ = _make_student(client)
    r1 = client.post(f"/api/courses/{course['id']}/enroll", headers=auth_header(stud_token))
    r2 = client.post(f"/api/courses/{course['id']}/enroll", headers=auth_header(stud_token))
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r1.get_json()["id"] == r2.get_json()["id"]


def test_instructor_cannot_enroll(client):
    token, _ = _make_instructor(client)
    course = client.post("/api/courses", json={"title": "C"}, headers=auth_header(token)).get_json()
    r = client.post(f"/api/courses/{course['id']}/enroll", headers=auth_header(token))
    assert r.status_code == 403


def test_progress_requires_enrollment(client):
    token, _ = _make_instructor(client)
    course = client.post("/api/courses", json={"title": "C"}, headers=auth_header(token)).get_json()
    stud_token, _ = _make_student(client)
    r = client.get(f"/api/courses/{course['id']}/progress", headers=auth_header(stud_token))
    assert r.status_code == 403
