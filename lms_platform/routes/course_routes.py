from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..decorators import require_auth, require_role

bp = Blueprint("course_routes", __name__, url_prefix="/api/courses")


def _course_owned_by(conn, course_id: int, user_id: int):
    """Return the course row if it exists and belongs to user_id, else None."""
    course = db.get_course(conn, course_id)
    if course is None or course["instructor_id"] != user_id:
        return None
    return course


@bp.get("")
@require_auth
def list_courses():
    """All authenticated users see the full catalogue. ?mine=1 restricts
    an instructor's view to the courses they own."""
    conn = current_app.config["DB_CONN"]
    if request.args.get("mine") == "1":
        if g.role != "instructor":
            return jsonify(error="only instructors can filter by ?mine=1"), 403
        courses = db.list_courses(conn, instructor_id=g.user_id)
    else:
        courses = db.list_courses(conn)
    return jsonify(courses=[db.row_to_dict(c) for c in courses])


@bp.post("")
@require_auth
@require_role("instructor")
def create_course():
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    description = (body.get("description") or "").strip()
    if not title:
        return jsonify(error="title is required"), 400

    conn = current_app.config["DB_CONN"]
    with db.transaction(conn):
        course_id = db.create_course(conn, title, description, g.user_id)
    return jsonify(db.row_to_dict(db.get_course(conn, course_id))), 201


@bp.get("/<int:course_id>")
@require_auth
def get_course(course_id):
    conn = current_app.config["DB_CONN"]
    course = db.get_course(conn, course_id)
    if course is None:
        return jsonify(error="course not found"), 404
    return jsonify(db.row_to_dict(course))


@bp.patch("/<int:course_id>")
@require_auth
@require_role("instructor")
def update_course(course_id):
    conn = current_app.config["DB_CONN"]
    course = _course_owned_by(conn, course_id, g.user_id)
    if course is None:
        return jsonify(error="course not found"), 404

    body = request.get_json(silent=True) or {}
    fields = {}
    if "title" in body:
        title = (body.get("title") or "").strip()
        if not title:
            return jsonify(error="title cannot be blank"), 400
        fields["title"] = title
    if "description" in body:
        fields["description"] = (body.get("description") or "").strip()

    if fields:
        with db.transaction(conn):
            db.update_course(conn, course_id, **fields)
    return jsonify(db.row_to_dict(db.get_course(conn, course_id)))


@bp.delete("/<int:course_id>")
@require_auth
@require_role("instructor")
def delete_course(course_id):
    """Owner-only. Cascades to the course's lessons, enrollments, and
    progress rows via ON DELETE CASCADE foreign keys."""
    conn = current_app.config["DB_CONN"]
    course = _course_owned_by(conn, course_id, g.user_id)
    if course is None:
        return jsonify(error="course not found"), 404

    with db.transaction(conn):
        db.delete_course(conn, course_id)
    return jsonify(deleted=True, course_id=course_id)


@bp.post("/<int:course_id>/enroll")
@require_auth
@require_role("student")
def enroll(course_id):
    conn = current_app.config["DB_CONN"]
    course = db.get_course(conn, course_id)
    if course is None:
        return jsonify(error="course not found"), 404

    existing = db.get_enrollment(conn, g.user_id, course_id)
    if existing is not None:
        return jsonify(db.row_to_dict(existing)), 200

    with db.transaction(conn):
        enrollment_id = db.create_enrollment(conn, g.user_id, course_id)
    return jsonify(db.row_to_dict(db.get_enrollment_by_id(conn, enrollment_id))), 201


@bp.get("/<int:course_id>/progress")
@require_auth
@require_role("student")
def course_progress(course_id):
    conn = current_app.config["DB_CONN"]
    course = db.get_course(conn, course_id)
    if course is None:
        return jsonify(error="course not found"), 404

    enrollment = db.get_enrollment(conn, g.user_id, course_id)
    if enrollment is None:
        return jsonify(error="not enrolled in this course"), 403

    lessons = db.list_lessons_for_course(conn, course_id)
    completed_ids = db.list_completed_lesson_ids(conn, enrollment["id"])
    lesson_progress = [
        {"lesson_id": l["id"], "title": l["title"], "completed": l["id"] in completed_ids}
        for l in lessons
    ]
    total = len(lessons)
    completed_count = sum(1 for lp in lesson_progress if lp["completed"])
    percent = round(100 * completed_count / total, 1) if total else 0.0
    return jsonify(
        course_id=course_id,
        lessons=lesson_progress,
        completed_count=completed_count,
        total_lessons=total,
        percent_complete=percent,
    )
