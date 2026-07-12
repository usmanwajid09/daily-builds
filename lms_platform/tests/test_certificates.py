from lms_platform.tests.conftest import auth_header, signup


def _instructor(client, email="prof@example.com"):
    return signup(client, email, "password123", "instructor")


def _student(client, email="stud@example.com"):
    return signup(client, email, "password123", "student")


def _make_course(client, token):
    r = client.post("/api/courses", json={"title": "C1"}, headers=auth_header(token))
    return r.get_json()["id"]


def test_certificate_denied_without_enrollment(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    r = client.post(f"/api/courses/{course_id}/certificate", headers=auth_header(stoken))
    assert r.status_code == 403


def test_certificate_denied_with_incomplete_lessons(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    client.post(f"/api/courses/{course_id}/lessons",
                json={"title": "L1", "content_type": "text", "content": "x"},
                headers=auth_header(itoken))
    stoken, _ = _student(client)
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))

    r = client.post(f"/api/courses/{course_id}/certificate", headers=auth_header(stoken))
    assert r.status_code == 403
    body = r.get_json()
    assert any("lesson" in reason for reason in body["reasons"])


def test_certificate_denied_when_quiz_not_passed(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))
    client.post(f"/api/courses/{course_id}/quizzes",
                json={"title": "Q1", "passing_score": 50, "questions": [
                    {"question_text": "2+2?", "options": [
                        {"option_text": "4", "is_correct": True},
                        {"option_text": "5", "is_correct": False},
                    ]},
                ]}, headers=auth_header(itoken))

    r = client.post(f"/api/courses/{course_id}/certificate", headers=auth_header(stoken))
    assert r.status_code == 403
    assert any("not yet passed" in reason for reason in r.get_json()["reasons"])


def test_certificate_denied_when_assignment_not_submitted(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))
    client.post(f"/api/courses/{course_id}/assignments", json={"title": "HW1"},
                headers=auth_header(itoken))

    r = client.post(f"/api/courses/{course_id}/certificate", headers=auth_header(stoken))
    assert r.status_code == 403
    assert any("not yet submitted" in reason for reason in r.get_json()["reasons"])


def test_certificate_issued_when_fully_eligible_empty_course(client):
    """A course with zero lessons/quizzes/assignments is trivially complete."""
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))

    r = client.post(f"/api/courses/{course_id}/certificate", headers=auth_header(stoken))
    assert r.status_code == 201, r.get_json()
    assert "verification_code" in r.get_json()


def test_certificate_reissue_is_idempotent(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))

    r1 = client.post(f"/api/courses/{course_id}/certificate", headers=auth_header(stoken))
    r2 = client.post(f"/api/courses/{course_id}/certificate", headers=auth_header(stoken))
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r1.get_json()["verification_code"] == r2.get_json()["verification_code"]


def test_certificate_public_verification(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client, "grad@example.com")
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))
    r = client.post(f"/api/courses/{course_id}/certificate", headers=auth_header(stoken))
    code = r.get_json()["verification_code"]

    # No Authorization header at all.
    r = client.get(f"/api/certificates/{code}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["student_email"] == "grad@example.com"
    assert body["course_title"] == "C1"


def test_verify_unknown_certificate_code_404(client):
    r = client.get("/api/certificates/does-not-exist")
    assert r.status_code == 404


def test_list_my_certificates(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))
    client.post(f"/api/courses/{course_id}/certificate", headers=auth_header(stoken))

    r = client.get("/api/me/certificates", headers=auth_header(stoken))
    assert r.status_code == 200
    certs = r.get_json()["certificates"]
    assert len(certs) == 1
    assert certs[0]["course_title"] == "C1"
