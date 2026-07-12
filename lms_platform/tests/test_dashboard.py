from lms_platform.tests.conftest import auth_header, signup


def _instructor(client, email="prof@example.com"):
    return signup(client, email, "password123", "instructor")


def _student(client, email="stud@example.com"):
    return signup(client, email, "password123", "student")


def _make_course(client, token):
    r = client.post("/api/courses", json={"title": "C1"}, headers=auth_header(token))
    return r.get_json()["id"]


def test_student_dashboard_shape_and_progress(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    r = client.post(f"/api/courses/{course_id}/lessons",
                     json={"title": "L1", "content_type": "text", "content": "x"},
                     headers=auth_header(itoken))
    lesson_id = r.get_json()["id"]

    stoken, _ = _student(client)
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))

    r = client.get("/api/me/dashboard", headers=auth_header(stoken))
    assert r.status_code == 200
    body = r.get_json()
    assert body["role"] == "student"
    assert body["courses_enrolled"] == 1
    course = body["courses"][0]
    assert course["lesson_percent_complete"] == 0.0
    assert course["certificate_earned"] is False

    client.post(f"/api/lessons/{lesson_id}/complete", headers=auth_header(stoken))
    r = client.get("/api/me/dashboard", headers=auth_header(stoken))
    course = r.get_json()["courses"][0]
    assert course["lesson_percent_complete"] == 100.0


def test_instructor_dashboard_counts_students_and_ungraded(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    r = client.post(f"/api/courses/{course_id}/assignments", json={"title": "HW1"},
                     headers=auth_header(itoken))
    assignment_id = r.get_json()["id"]

    s1, sid1 = _student(client, "s1@example.com")
    s2, sid2 = _student(client, "s2@example.com")
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(s1))
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(s2))
    client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "a"},
                headers=auth_header(s1))
    client.post(f"/api/assignments/{assignment_id}/submit", json={"content": "b"},
                headers=auth_header(s2))

    r = client.get("/api/me/dashboard", headers=auth_header(itoken))
    body = r.get_json()
    assert body["role"] == "instructor"
    course = body["courses"][0]
    assert course["student_count"] == 2
    assert course["ungraded_submissions"] == 2
    assert body["ungraded_submissions_total"] == 2

    client.patch(f"/api/assignments/{assignment_id}/submissions/{sid1}/grade",
                 json={"grade": 90}, headers=auth_header(itoken))
    r = client.get("/api/me/dashboard", headers=auth_header(itoken))
    course = r.get_json()["courses"][0]
    assert course["ungraded_submissions"] == 1


def test_instructor_dashboard_quiz_pass_rate_uses_best_attempt_per_student(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    r = client.post(f"/api/courses/{course_id}/quizzes",
                     json={"title": "Q1", "passing_score": 50, "questions": [
                         {"question_text": "2+2?", "options": [
                             {"option_text": "4", "is_correct": True},
                             {"option_text": "5", "is_correct": False},
                         ]},
                     ]}, headers=auth_header(itoken))
    quiz = r.get_json()
    qid = quiz["questions"][0]["id"]
    correct_opt = [o["id"] for o in quiz["questions"][0]["options"] if o["is_correct"]][0]
    wrong_opt = [o["id"] for o in quiz["questions"][0]["options"] if not o["is_correct"]][0]

    stoken, _ = _student(client)
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))
    # First attempt fails...
    client.post(f"/api/quizzes/{quiz['id']}/attempts",
                json={"answers": [{"question_id": qid, "option_id": wrong_opt}]},
                headers=auth_header(stoken))
    # ...second attempt passes. Pass rate should reflect the best attempt (100%), not 50%.
    client.post(f"/api/quizzes/{quiz['id']}/attempts",
                json={"answers": [{"question_id": qid, "option_id": correct_opt}]},
                headers=auth_header(stoken))

    r = client.get("/api/me/dashboard", headers=auth_header(itoken))
    course = r.get_json()["courses"][0]
    assert course["avg_quiz_pass_rate"] == 100.0
