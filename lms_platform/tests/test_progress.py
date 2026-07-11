from lms_platform.tests.conftest import auth_header, signup


def _setup_course_with_lessons(client, n=3):
    token, _ = signup(client, "prof@example.com", "password123", "instructor")
    course = client.post("/api/courses", json={"title": "C"}, headers=auth_header(token)).get_json()
    lessons = []
    for i in range(n):
        lesson = client.post(
            f"/api/courses/{course['id']}/lessons",
            json={"title": f"L{i}", "content_type": "text", "content": "x"},
            headers=auth_header(token),
        ).get_json()
        lessons.append(lesson)
    return token, course, lessons


def test_progress_zero_percent_before_completing_anything(client):
    _, course, _ = _setup_course_with_lessons(client, 2)
    stud_token, _ = signup(client, "stud@example.com", "password123", "student")
    client.post(f"/api/courses/{course['id']}/enroll", headers=auth_header(stud_token))
    r = client.get(f"/api/courses/{course['id']}/progress", headers=auth_header(stud_token))
    assert r.get_json()["percent_complete"] == 0.0


def test_progress_updates_as_lessons_complete(client):
    _, course, lessons = _setup_course_with_lessons(client, 4)
    stud_token, _ = signup(client, "stud@example.com", "password123", "student")
    client.post(f"/api/courses/{course['id']}/enroll", headers=auth_header(stud_token))

    client.post(f"/api/lessons/{lessons[0]['id']}/complete", headers=auth_header(stud_token))
    r = client.get(f"/api/courses/{course['id']}/progress", headers=auth_header(stud_token))
    assert r.get_json()["percent_complete"] == 25.0

    client.post(f"/api/lessons/{lessons[1]['id']}/complete", headers=auth_header(stud_token))
    r = client.get(f"/api/courses/{course['id']}/progress", headers=auth_header(stud_token))
    assert r.get_json()["percent_complete"] == 50.0


def test_completing_a_lesson_twice_is_idempotent(client):
    _, course, lessons = _setup_course_with_lessons(client, 2)
    stud_token, _ = signup(client, "stud@example.com", "password123", "student")
    client.post(f"/api/courses/{course['id']}/enroll", headers=auth_header(stud_token))

    r1 = client.post(f"/api/lessons/{lessons[0]['id']}/complete", headers=auth_header(stud_token))
    r2 = client.post(f"/api/lessons/{lessons[0]['id']}/complete", headers=auth_header(stud_token))
    assert r1.status_code == 200
    assert r2.status_code == 200
    r = client.get(f"/api/courses/{course['id']}/progress", headers=auth_header(stud_token))
    assert r.get_json()["percent_complete"] == 50.0


def test_cannot_complete_lesson_without_enrollment(client):
    _, course, lessons = _setup_course_with_lessons(client, 1)
    stud_token, _ = signup(client, "stud@example.com", "password123", "student")
    r = client.post(f"/api/lessons/{lessons[0]['id']}/complete", headers=auth_header(stud_token))
    assert r.status_code == 403


def test_instructor_cannot_mark_lesson_complete(client):
    token, course, lessons = _setup_course_with_lessons(client, 1)
    r = client.post(f"/api/lessons/{lessons[0]['id']}/complete", headers=auth_header(token))
    assert r.status_code == 403


def test_my_enrollments_reflects_progress(client):
    _, course, lessons = _setup_course_with_lessons(client, 2)
    stud_token, _ = signup(client, "stud@example.com", "password123", "student")
    client.post(f"/api/courses/{course['id']}/enroll", headers=auth_header(stud_token))
    client.post(f"/api/lessons/{lessons[0]['id']}/complete", headers=auth_header(stud_token))

    r = client.get("/api/me/enrollments", headers=auth_header(stud_token))
    enrollments = r.get_json()["enrollments"]
    assert len(enrollments) == 1
    assert enrollments[0]["completed_count"] == 1
    assert enrollments[0]["total_lessons"] == 2
    assert enrollments[0]["percent_complete"] == 50.0
    assert enrollments[0]["course_title"] == "C"


def test_progress_with_zero_lessons_is_zero_not_a_crash(client):
    token, _ = signup(client, "prof@example.com", "password123", "instructor")
    course = client.post("/api/courses", json={"title": "Empty"}, headers=auth_header(token)).get_json()
    stud_token, _ = signup(client, "stud@example.com", "password123", "student")
    client.post(f"/api/courses/{course['id']}/enroll", headers=auth_header(stud_token))
    r = client.get(f"/api/courses/{course['id']}/progress", headers=auth_header(stud_token))
    assert r.status_code == 200
    assert r.get_json()["percent_complete"] == 0.0
