from lms_platform.tests.conftest import auth_header, signup


def _instructor(client, email="prof@example.com"):
    return signup(client, email, "password123", "instructor")


def _student(client, email="stud@example.com"):
    return signup(client, email, "password123", "student")


def _make_course(client, token):
    r = client.post("/api/courses", json={"title": "C1"}, headers=auth_header(token))
    return r.get_json()["id"]


VALID_QUIZ_BODY = {
    "title": "Quiz 1",
    "passing_score": 50,
    "questions": [
        {"question_text": "2+2?", "points": 1, "options": [
            {"option_text": "4", "is_correct": True},
            {"option_text": "5", "is_correct": False},
        ]},
        {"question_text": "Capital of France?", "points": 1, "options": [
            {"option_text": "Paris", "is_correct": True},
            {"option_text": "London", "is_correct": False},
        ]},
    ],
}


def test_instructor_can_create_quiz_with_questions_and_options(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    r = client.post(f"/api/courses/{course_id}/quizzes", json=VALID_QUIZ_BODY,
                     headers=auth_header(token))
    assert r.status_code == 201, r.get_json()
    body = r.get_json()
    assert body["title"] == "Quiz 1"
    assert len(body["questions"]) == 2
    assert all("is_correct" in o for q in body["questions"] for o in q["options"])


def test_student_cannot_create_quiz(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    stoken, _ = _student(client)
    r = client.post(f"/api/courses/{course_id}/quizzes", json=VALID_QUIZ_BODY,
                     headers=auth_header(stoken))
    assert r.status_code == 403


def test_non_owner_instructor_cannot_create_quiz(client):
    token, _ = _instructor(client, "a@example.com")
    course_id = _make_course(client, token)
    other_token, _ = _instructor(client, "b@example.com")
    r = client.post(f"/api/courses/{course_id}/quizzes", json=VALID_QUIZ_BODY,
                     headers=auth_header(other_token))
    assert r.status_code == 404


def test_quiz_requires_at_least_one_question(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    r = client.post(f"/api/courses/{course_id}/quizzes",
                     json={"title": "Empty", "questions": []}, headers=auth_header(token))
    assert r.status_code == 400


def test_quiz_question_requires_two_options(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    body = {"title": "Q", "questions": [
        {"question_text": "Only one option?", "options": [
            {"option_text": "only", "is_correct": True},
        ]},
    ]}
    r = client.post(f"/api/courses/{course_id}/quizzes", json=body, headers=auth_header(token))
    assert r.status_code == 400


def test_quiz_question_requires_exactly_one_correct_option(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    body = {"title": "Q", "questions": [
        {"question_text": "Both correct?", "options": [
            {"option_text": "a", "is_correct": True},
            {"option_text": "b", "is_correct": True},
        ]},
    ]}
    r = client.post(f"/api/courses/{course_id}/quizzes", json=body, headers=auth_header(token))
    assert r.status_code == 400

    body["questions"][0]["options"] = [
        {"option_text": "a", "is_correct": False},
        {"option_text": "b", "is_correct": False},
    ]
    r = client.post(f"/api/courses/{course_id}/quizzes", json=body, headers=auth_header(token))
    assert r.status_code == 400


def test_student_view_hides_correct_answers(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    r = client.post(f"/api/courses/{course_id}/quizzes", json=VALID_QUIZ_BODY,
                     headers=auth_header(token))
    quiz_id = r.get_json()["id"]

    stoken, _ = _student(client)
    r = client.get(f"/api/quizzes/{quiz_id}", headers=auth_header(stoken))
    assert r.status_code == 200
    body = r.get_json()
    assert all("is_correct" not in o for q in body["questions"] for o in q["options"])


def test_instructor_view_shows_correct_answers(client):
    token, _ = _instructor(client)
    course_id = _make_course(client, token)
    r = client.post(f"/api/courses/{course_id}/quizzes", json=VALID_QUIZ_BODY,
                     headers=auth_header(token))
    quiz_id = r.get_json()["id"]
    r = client.get(f"/api/quizzes/{quiz_id}", headers=auth_header(token))
    body = r.get_json()
    assert all("is_correct" in o for q in body["questions"] for o in q["options"])


def _enroll_and_get_quiz(client, itoken, stoken, course_id, passing_score=50):
    body = dict(VALID_QUIZ_BODY, passing_score=passing_score)
    r = client.post(f"/api/courses/{course_id}/quizzes", json=body, headers=auth_header(itoken))
    quiz = r.get_json()
    client.post(f"/api/courses/{course_id}/enroll", headers=auth_header(stoken))
    return quiz


def test_submitting_attempt_requires_enrollment(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    r = client.post(f"/api/courses/{course_id}/quizzes", json=VALID_QUIZ_BODY,
                     headers=auth_header(itoken))
    quiz = r.get_json()
    stoken, _ = _student(client)
    answers = [{"question_id": q["id"],
                "option_id": [o["id"] for o in q["options"]][0]} for q in quiz["questions"]]
    r = client.post(f"/api/quizzes/{quiz['id']}/attempts", json={"answers": answers},
                     headers=auth_header(stoken))
    assert r.status_code == 403


def test_all_correct_answers_pass(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    quiz = _enroll_and_get_quiz(client, itoken, stoken, course_id, passing_score=50)
    answers = [
        {"question_id": q["id"],
         "option_id": [o["id"] for o in q["options"] if o["is_correct"]][0]}
        for q in quiz["questions"]
    ]
    r = client.post(f"/api/quizzes/{quiz['id']}/attempts", json={"answers": answers},
                     headers=auth_header(stoken))
    assert r.status_code == 201, r.get_json()
    body = r.get_json()
    assert body["percent"] == 100.0
    assert body["passed"] is True


def test_all_wrong_answers_fail_and_score_zero(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    quiz = _enroll_and_get_quiz(client, itoken, stoken, course_id, passing_score=50)
    answers = [
        {"question_id": q["id"],
         "option_id": [o["id"] for o in q["options"] if not o["is_correct"]][0]}
        for q in quiz["questions"]
    ]
    r = client.post(f"/api/quizzes/{quiz['id']}/attempts", json={"answers": answers},
                     headers=auth_header(stoken))
    body = r.get_json()
    assert body["percent"] == 0.0
    assert body["passed"] is False


def test_partial_credit_respects_passing_score_threshold(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    quiz = _enroll_and_get_quiz(client, itoken, stoken, course_id, passing_score=60)
    q0, q1 = quiz["questions"]
    answers = [
        {"question_id": q0["id"], "option_id": [o["id"] for o in q0["options"] if o["is_correct"]][0]},
        {"question_id": q1["id"], "option_id": [o["id"] for o in q1["options"] if not o["is_correct"]][0]},
    ]
    r = client.post(f"/api/quizzes/{quiz['id']}/attempts", json={"answers": answers},
                     headers=auth_header(stoken))
    body = r.get_json()
    assert body["percent"] == 50.0
    assert body["passed"] is False  # 50 < passing_score of 60


def test_unanswered_question_scored_as_wrong_not_an_error(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    quiz = _enroll_and_get_quiz(client, itoken, stoken, course_id, passing_score=50)
    q0 = quiz["questions"][0]
    answers = [
        {"question_id": q0["id"], "option_id": [o["id"] for o in q0["options"] if o["is_correct"]][0]},
    ]
    r = client.post(f"/api/quizzes/{quiz['id']}/attempts", json={"answers": answers},
                     headers=auth_header(stoken))
    assert r.status_code == 201
    body = r.get_json()
    assert body["percent"] == 50.0


def test_option_must_belong_to_its_question(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    quiz = _enroll_and_get_quiz(client, itoken, stoken, course_id)
    q0, q1 = quiz["questions"]
    # Swap: answer question 0 with an option belonging to question 1.
    wrong_option_id = q1["options"][0]["id"]
    r = client.post(f"/api/quizzes/{quiz['id']}/attempts",
                     json={"answers": [{"question_id": q0["id"], "option_id": wrong_option_id}]},
                     headers=auth_header(stoken))
    assert r.status_code == 400


def test_duplicate_answer_for_same_question_rejected(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    quiz = _enroll_and_get_quiz(client, itoken, stoken, course_id)
    q0 = quiz["questions"][0]
    opt = q0["options"][0]["id"]
    r = client.post(f"/api/quizzes/{quiz['id']}/attempts",
                     json={"answers": [{"question_id": q0["id"], "option_id": opt},
                                        {"question_id": q0["id"], "option_id": opt}]},
                     headers=auth_header(stoken))
    assert r.status_code == 400


def test_instructor_can_list_attempts_for_their_quiz(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client, "s1@example.com")
    quiz = _enroll_and_get_quiz(client, itoken, stoken, course_id)
    answers = [{"question_id": q["id"],
                "option_id": [o["id"] for o in q["options"] if o["is_correct"]][0]}
               for q in quiz["questions"]]
    client.post(f"/api/quizzes/{quiz['id']}/attempts", json={"answers": answers},
                headers=auth_header(stoken))
    r = client.get(f"/api/quizzes/{quiz['id']}/attempts", headers=auth_header(itoken))
    assert r.status_code == 200
    attempts = r.get_json()["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["student_email"] == "s1@example.com"


def test_student_can_list_their_own_attempts(client):
    itoken, _ = _instructor(client)
    course_id = _make_course(client, itoken)
    stoken, _ = _student(client)
    quiz = _enroll_and_get_quiz(client, itoken, stoken, course_id)
    answers = [{"question_id": q["id"],
                "option_id": [o["id"] for o in q["options"] if o["is_correct"]][0]}
               for q in quiz["questions"]]
    client.post(f"/api/quizzes/{quiz['id']}/attempts", json={"answers": answers},
                headers=auth_header(stoken))
    r = client.get("/api/me/quiz-attempts", headers=auth_header(stoken))
    assert r.status_code == 200
    assert len(r.get_json()["attempts"]) == 1


def test_delete_quiz_owner_only(client):
    itoken, _ = _instructor(client, "a@example.com")
    course_id = _make_course(client, itoken)
    r = client.post(f"/api/courses/{course_id}/quizzes", json=VALID_QUIZ_BODY,
                     headers=auth_header(itoken))
    quiz_id = r.get_json()["id"]

    other, _ = _instructor(client, "b@example.com")
    r = client.delete(f"/api/quizzes/{quiz_id}", headers=auth_header(other))
    assert r.status_code == 404

    r = client.delete(f"/api/quizzes/{quiz_id}", headers=auth_header(itoken))
    assert r.status_code == 200

    r = client.get(f"/api/quizzes/{quiz_id}", headers=auth_header(itoken))
    assert r.status_code == 404
