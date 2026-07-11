from flask import Blueprint, current_app, g, jsonify

from .. import db
from ..decorators import require_auth, require_role

bp = Blueprint("enrollment_routes", __name__, url_prefix="/api/me")


@bp.get("/enrollments")
@require_auth
@require_role("student")
def my_enrollments():
    conn = current_app.config["DB_CONN"]
    enrollments = db.list_enrollments_for_student(conn, g.user_id)

    result = []
    for e in enrollments:
        course = db.get_course(conn, e["course_id"])
        lessons = db.list_lessons_for_course(conn, e["course_id"])
        completed_ids = db.list_completed_lesson_ids(conn, e["id"])
        total = len(lessons)
        completed_count = sum(1 for l in lessons if l["id"] in completed_ids)
        percent = round(100 * completed_count / total, 1) if total else 0.0
        result.append({
            "enrollment_id": e["id"],
            "course_id": e["course_id"],
            "course_title": course["title"] if course else None,
            "enrolled_at": e["enrolled_at"],
            "completed_count": completed_count,
            "total_lessons": total,
            "percent_complete": percent,
        })
    return jsonify(enrollments=result)
