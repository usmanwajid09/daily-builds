"""Assignment CRUD (instructor-authored, scoped under a course), student
submission (resubmission allowed and clears any prior grade), and
manual instructor grading."""
from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..decorators import require_auth, require_role

bp = Blueprint("assignment_routes", __name__, url_prefix="/api")


def _course_owned_by(conn, course_id: int, user_id: int):
    course = db.get_course(conn, course_id)
    if course is None or course["instructor_id"] != user_id:
        return None
    return course


def _assignment_with_course(conn, assignment_id: int):
    assignment = db.get_assignment(conn, assignment_id)
    if assignment is None:
        return None, None
    course = db.get_course(conn, assignment["course_id"])
    return assignment, course


@bp.get("/courses/<int:course_id>/assignments")
@require_auth
def list_assignments(course_id):
    conn = current_app.config["DB_CONN"]
    course = db.get_course(conn, course_id)
    if course is None:
        return jsonify(error="course not found"), 404
    assignments = db.list_assignments_for_course(conn, course_id)
    return jsonify(assignments=[db.row_to_dict(a) for a in assignments])


@bp.post("/courses/<int:course_id>/assignments")
@require_auth
@require_role("instructor")
def create_assignment(course_id):
    conn = current_app.config["DB_CONN"]
    course = _course_owned_by(conn, course_id, g.user_id)
    if course is None:
        return jsonify(error="course not found"), 404

    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    if not title:
        return jsonify(error="title is required"), 400
    description = (body.get("description") or "").strip()
    try:
        max_points = float(body.get("max_points", 100.0))
    except (TypeError, ValueError):
        return jsonify(error="max_points must be a number"), 400
    if max_points <= 0:
        return jsonify(error="max_points must be positive"), 400
    due_at = body.get("due_at")
    if due_at is not None and not isinstance(due_at, str):
        return jsonify(error="due_at must be an ISO date string or null"), 400

    with db.transaction(conn):
        assignment_id = db.create_assignment(conn, course_id, title, description,
                                              max_points, due_at)
    return jsonify(db.row_to_dict(db.get_assignment(conn, assignment_id))), 201


@bp.get("/assignments/<int:assignment_id>")
@require_auth
def get_assignment(assignment_id):
    conn = current_app.config["DB_CONN"]
    assignment = db.get_assignment(conn, assignment_id)
    if assignment is None:
        return jsonify(error="assignment not found"), 404
    return jsonify(db.row_to_dict(assignment))


@bp.delete("/assignments/<int:assignment_id>")
@require_auth
@require_role("instructor")
def delete_assignment(assignment_id):
    conn = current_app.config["DB_CONN"]
    assignment, course = _assignment_with_course(conn, assignment_id)
    if assignment is None or course["instructor_id"] != g.user_id:
        return jsonify(error="assignment not found"), 404
    with db.transaction(conn):
        db.delete_assignment(conn, assignment_id)
    return jsonify(deleted=True, assignment_id=assignment_id)


@bp.post("/assignments/<int:assignment_id>/submit")
@require_auth
@require_role("student")
def submit_assignment(assignment_id):
    conn = current_app.config["DB_CONN"]
    assignment, course = _assignment_with_course(conn, assignment_id)
    if assignment is None:
        return jsonify(error="assignment not found"), 404

    enrollment = db.get_enrollment(conn, g.user_id, course["id"])
    if enrollment is None:
        return jsonify(error="not enrolled in this assignment's course"), 403

    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        return jsonify(error="content is required"), 400

    with db.transaction(conn):
        submission_id = db.upsert_submission(conn, assignment_id, g.user_id, content)
    return jsonify(db.row_to_dict(db.get_submission_by_id(conn, submission_id))), 200


@bp.get("/assignments/<int:assignment_id>/submissions")
@require_auth
@require_role("instructor")
def list_submissions(assignment_id):
    conn = current_app.config["DB_CONN"]
    assignment, course = _assignment_with_course(conn, assignment_id)
    if assignment is None or course["instructor_id"] != g.user_id:
        return jsonify(error="assignment not found"), 404
    submissions = db.list_submissions_for_assignment(conn, assignment_id)
    out = []
    for s in submissions:
        student = db.get_user(conn, s["student_id"])
        row = db.row_to_dict(s)
        row["student_email"] = student["email"] if student else None
        out.append(row)
    return jsonify(submissions=out)


@bp.get("/me/submissions")
@require_auth
@require_role("student")
def my_submissions():
    conn = current_app.config["DB_CONN"]
    assignment_id = request.args.get("assignment_id", type=int)
    if assignment_id is not None:
        submission = db.get_submission(conn, assignment_id, g.user_id)
        return jsonify(submissions=[db.row_to_dict(submission)] if submission else [])
    # No dedicated "all submissions for a student across courses" query
    # exists in db.py (submissions are only looked up per-assignment) --
    # the dashboard covers the aggregate view instead.
    return jsonify(error="assignment_id query parameter is required"), 400


@bp.patch("/assignments/<int:assignment_id>/submissions/<int:student_id>/grade")
@require_auth
@require_role("instructor")
def grade_submission(assignment_id, student_id):
    conn = current_app.config["DB_CONN"]
    assignment, course = _assignment_with_course(conn, assignment_id)
    if assignment is None or course["instructor_id"] != g.user_id:
        return jsonify(error="assignment not found"), 404

    submission = db.get_submission(conn, assignment_id, student_id)
    if submission is None:
        return jsonify(error="no submission from this student for this assignment"), 404

    body = request.get_json(silent=True) or {}
    if "grade" not in body:
        return jsonify(error="grade is required"), 400
    try:
        grade = float(body["grade"])
    except (TypeError, ValueError):
        return jsonify(error="grade must be a number"), 400
    if not (0 <= grade <= assignment["max_points"]):
        return jsonify(error=f"grade must be between 0 and {assignment['max_points']}"), 400
    feedback = (body.get("feedback") or "").strip()

    with db.transaction(conn):
        db.grade_submission(conn, submission["id"], grade, feedback)
    return jsonify(db.row_to_dict(db.get_submission_by_id(conn, submission["id"])))
