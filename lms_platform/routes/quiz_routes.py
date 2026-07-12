"""Quiz CRUD (instructor-authored, scoped under a course) + student
attempts (auto-graded against the correct-option flags set at
creation time)."""
from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..decorators import require_auth, require_role
from ..grading import grade_quiz

bp = Blueprint("quiz_routes", __name__, url_prefix="/api")


def _course_owned_by(conn, course_id: int, user_id: int):
    course = db.get_course(conn, course_id)
    if course is None or course["instructor_id"] != user_id:
        return None
    return course


def _quiz_with_course(conn, quiz_id: int):
    quiz = db.get_quiz(conn, quiz_id)
    if quiz is None:
        return None, None
    course = db.get_course(conn, quiz["course_id"])
    return quiz, course


def _serialize_quiz(conn, quiz, reveal_correct: bool):
    questions = db.list_questions_for_quiz(conn, quiz["id"])
    q_out = []
    for q in questions:
        options = db.list_options_for_question(conn, q["id"])
        opt_out = []
        for o in options:
            item = {"id": o["id"], "option_text": o["option_text"], "order_index": o["order_index"]}
            if reveal_correct:
                item["is_correct"] = bool(o["is_correct"])
            opt_out.append(item)
        q_out.append({
            "id": q["id"], "question_text": q["question_text"],
            "order_index": q["order_index"], "points": q["points"], "options": opt_out,
        })
    return {
        "id": quiz["id"], "course_id": quiz["course_id"], "title": quiz["title"],
        "passing_score": quiz["passing_score"], "questions": q_out,
    }


@bp.get("/courses/<int:course_id>/quizzes")
@require_auth
def list_quizzes(course_id):
    conn = current_app.config["DB_CONN"]
    course = db.get_course(conn, course_id)
    if course is None:
        return jsonify(error="course not found"), 404
    quizzes = db.list_quizzes_for_course(conn, course_id)
    is_owner = g.role == "instructor" and course["instructor_id"] == g.user_id
    out = []
    for quiz in quizzes:
        questions = db.list_questions_for_quiz(conn, quiz["id"])
        out.append({
            "id": quiz["id"], "title": quiz["title"], "passing_score": quiz["passing_score"],
            "question_count": len(questions),
        })
    return jsonify(quizzes=out, can_see_answers=is_owner)


@bp.post("/courses/<int:course_id>/quizzes")
@require_auth
@require_role("instructor")
def create_quiz(course_id):
    conn = current_app.config["DB_CONN"]
    course = _course_owned_by(conn, course_id, g.user_id)
    if course is None:
        return jsonify(error="course not found"), 404

    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    if not title:
        return jsonify(error="title is required"), 400
    try:
        passing_score = float(body.get("passing_score", 70.0))
    except (TypeError, ValueError):
        return jsonify(error="passing_score must be a number"), 400
    if not (0 <= passing_score <= 100):
        return jsonify(error="passing_score must be between 0 and 100"), 400

    questions = body.get("questions")
    if not isinstance(questions, list) or not questions:
        return jsonify(error="at least one question is required"), 400

    for i, q in enumerate(questions):
        if not isinstance(q, dict) or not (q.get("question_text") or "").strip():
            return jsonify(error=f"question {i}: question_text is required"), 400
        options = q.get("options")
        if not isinstance(options, list) or len(options) < 2:
            return jsonify(error=f"question {i}: at least two options are required"), 400
        for o in options:
            if not isinstance(o, dict) or not (o.get("option_text") or "").strip():
                return jsonify(error=f"question {i}: every option needs option_text"), 400
        correct_count = sum(1 for o in options if bool(o.get("is_correct")))
        if correct_count != 1:
            return jsonify(
                error=f"question {i}: exactly one option must be marked is_correct "
                      f"(found {correct_count})"
            ), 400

    with db.transaction(conn):
        quiz_id = db.create_quiz(conn, course_id, title, passing_score)
        for qi, q in enumerate(questions):
            points = q.get("points", 1.0)
            try:
                points = float(points)
            except (TypeError, ValueError):
                points = 1.0
            question_id = db.create_quiz_question(conn, quiz_id, q["question_text"].strip(),
                                                    qi, points)
            for oi, o in enumerate(q["options"]):
                db.create_quiz_option(conn, question_id, o["option_text"].strip(),
                                       bool(o.get("is_correct")), oi)

    quiz = db.get_quiz(conn, quiz_id)
    return jsonify(_serialize_quiz(conn, quiz, reveal_correct=True)), 201


@bp.get("/quizzes/<int:quiz_id>")
@require_auth
def get_quiz(quiz_id):
    conn = current_app.config["DB_CONN"]
    quiz, course = _quiz_with_course(conn, quiz_id)
    if quiz is None:
        return jsonify(error="quiz not found"), 404
    is_owner = g.role == "instructor" and course["instructor_id"] == g.user_id
    return jsonify(_serialize_quiz(conn, quiz, reveal_correct=is_owner))


@bp.delete("/quizzes/<int:quiz_id>")
@require_auth
@require_role("instructor")
def delete_quiz(quiz_id):
    conn = current_app.config["DB_CONN"]
    quiz, course = _quiz_with_course(conn, quiz_id)
    if quiz is None or course["instructor_id"] != g.user_id:
        return jsonify(error="quiz not found"), 404
    with db.transaction(conn):
        db.delete_quiz(conn, quiz_id)
    return jsonify(deleted=True, quiz_id=quiz_id)


@bp.post("/quizzes/<int:quiz_id>/attempts")
@require_auth
@require_role("student")
def submit_attempt(quiz_id):
    conn = current_app.config["DB_CONN"]
    quiz, course = _quiz_with_course(conn, quiz_id)
    if quiz is None:
        return jsonify(error="quiz not found"), 404

    enrollment = db.get_enrollment(conn, g.user_id, course["id"])
    if enrollment is None:
        return jsonify(error="not enrolled in this quiz's course"), 403

    body = request.get_json(silent=True) or {}
    answers = body.get("answers")
    if not isinstance(answers, list):
        return jsonify(error="answers must be a list of {question_id, option_id}"), 400

    questions = db.list_questions_for_quiz(conn, quiz_id)
    question_ids = {q["id"] for q in questions}
    correct_option_by_question = {}
    valid_option_ids_by_question = {}
    for q in questions:
        options = db.list_options_for_question(conn, q["id"])
        valid_option_ids_by_question[q["id"]] = {o["id"] for o in options}
        for o in options:
            if o["is_correct"]:
                correct_option_by_question[q["id"]] = o["id"]

    answers_by_question = {}
    for i, a in enumerate(answers):
        if not isinstance(a, dict) or "question_id" not in a or "option_id" not in a:
            return jsonify(error=f"answer {i}: each answer needs question_id and option_id"), 400
        qid, oid = a["question_id"], a["option_id"]
        if qid not in question_ids:
            return jsonify(error=f"answer {i}: question {qid} is not part of this quiz"), 400
        if qid in answers_by_question:
            return jsonify(error=f"answer {i}: duplicate answer for question {qid}"), 400
        if oid is not None and oid not in valid_option_ids_by_question[qid]:
            return jsonify(
                error=f"answer {i}: option {oid} does not belong to question {qid}"
            ), 400
        answers_by_question[qid] = oid

    result = grade_quiz(
        questions=[{"id": q["id"], "points": q["points"]} for q in questions],
        correct_option_by_question=correct_option_by_question,
        answers_by_question=answers_by_question,
        passing_score=quiz["passing_score"],
    )

    with db.transaction(conn):
        attempt_id = db.create_quiz_attempt(
            conn, quiz_id, g.user_id, result.score, result.total_points,
            result.percent, result.passed,
        )
        for qr in result.question_results:
            db.record_attempt_answer(conn, attempt_id, qr.question_id,
                                      qr.selected_option_id, qr.correct)

    return jsonify(
        attempt_id=attempt_id, score=result.score, total_points=result.total_points,
        percent=result.percent, passed=result.passed,
        breakdown=[
            {"question_id": qr.question_id, "correct": qr.correct, "points": qr.points,
             "awarded": qr.awarded}
            for qr in result.question_results
        ],
    ), 201


@bp.get("/quizzes/<int:quiz_id>/attempts")
@require_auth
@require_role("instructor")
def list_attempts(quiz_id):
    conn = current_app.config["DB_CONN"]
    quiz, course = _quiz_with_course(conn, quiz_id)
    if quiz is None or course["instructor_id"] != g.user_id:
        return jsonify(error="quiz not found"), 404
    attempts = db.list_attempts_for_quiz(conn, quiz_id)
    out = []
    for a in attempts:
        student = db.get_user(conn, a["student_id"])
        out.append({
            "id": a["id"], "student_id": a["student_id"],
            "student_email": student["email"] if student else None,
            "score": a["score"], "total_points": a["total_points"],
            "percent": a["percent"], "passed": bool(a["passed"]),
            "submitted_at": a["submitted_at"],
        })
    return jsonify(attempts=out)


@bp.get("/me/quiz-attempts")
@require_auth
@require_role("student")
def my_attempts():
    conn = current_app.config["DB_CONN"]
    quiz_id = request.args.get("quiz_id", type=int)
    attempts = db.list_attempts_for_student(conn, g.user_id, quiz_id=quiz_id)
    return jsonify(attempts=[db.row_to_dict(a) for a in attempts])
