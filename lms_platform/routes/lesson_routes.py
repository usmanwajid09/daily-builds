"""Lesson CRUD (scoped under a course) + a content-upload endpoint.

Lessons hold one of three content shapes, distinguished by
`content_type`: 'text' (content is the lesson body itself),
'video_url' (content is a URL), or 'file' (content is a path under
lms_platform/data/uploads/, set only by the /upload endpoint below --
you cannot set content_type='file' directly via create/update with
arbitrary content, since that would let a client claim an unuploaded
path exists).
"""
import os
import sqlite3
import uuid

from flask import Blueprint, current_app, g, jsonify, request
from werkzeug.utils import secure_filename

from .. import db
from ..decorators import require_auth, require_role

bp = Blueprint("lesson_routes", __name__, url_prefix="/api")

ALLOWED_UPLOAD_EXTENSIONS = {
    "pdf", "png", "jpg", "jpeg", "gif", "mp4", "mp3", "wav",
    "zip", "txt", "md", "docx", "pptx", "csv",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def _course_owned_by(conn, course_id: int, user_id: int):
    course = db.get_course(conn, course_id)
    if course is None or course["instructor_id"] != user_id:
        return None
    return course


def _lesson_with_course(conn, lesson_id: int):
    """Return (lesson_row, course_row) or (None, None) if the lesson
    doesn't exist (or its course was deleted out from under it, which
    ON DELETE CASCADE prevents anyway)."""
    lesson = db.get_lesson(conn, lesson_id)
    if lesson is None:
        return None, None
    course = db.get_course(conn, lesson["course_id"])
    return lesson, course


@bp.get("/courses/<int:course_id>/lessons")
@require_auth
def list_lessons(course_id):
    conn = current_app.config["DB_CONN"]
    course = db.get_course(conn, course_id)
    if course is None:
        return jsonify(error="course not found"), 404
    lessons = db.list_lessons_for_course(conn, course_id)
    return jsonify(lessons=[db.row_to_dict(l) for l in lessons])


@bp.post("/courses/<int:course_id>/lessons")
@require_auth
@require_role("instructor")
def create_lesson(course_id):
    conn = current_app.config["DB_CONN"]
    course = _course_owned_by(conn, course_id, g.user_id)
    if course is None:
        return jsonify(error="course not found"), 404

    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    content_type = body.get("content_type") or "text"
    content = body.get("content") or ""

    if not title:
        return jsonify(error="title is required"), 400
    if content_type not in ("text", "video_url"):
        return jsonify(
            error="content_type must be 'text' or 'video_url' here; "
                  "use POST /api/lessons/<id>/upload for file content"
        ), 400
    if content_type == "text" and not content.strip():
        return jsonify(error="content is required for a text lesson"), 400
    if content_type == "video_url" and not content.strip():
        return jsonify(error="content (the URL) is required for a video_url lesson"), 400

    order_index = body.get("order_index")
    try:
        with db.transaction(conn):
            if order_index is None:
                order_index = db.next_lesson_order_index(conn, course_id)
            lesson_id = db.create_lesson(conn, course_id, title, content_type, content, order_index)
    except sqlite3.IntegrityError:
        return jsonify(
            error=f"a lesson with order_index {order_index} already exists in this course"
        ), 409
    return jsonify(db.row_to_dict(db.get_lesson(conn, lesson_id))), 201


@bp.get("/lessons/<int:lesson_id>")
@require_auth
def get_lesson(lesson_id):
    conn = current_app.config["DB_CONN"]
    lesson = db.get_lesson(conn, lesson_id)
    if lesson is None:
        return jsonify(error="lesson not found"), 404
    return jsonify(db.row_to_dict(lesson))


@bp.patch("/lessons/<int:lesson_id>")
@require_auth
@require_role("instructor")
def update_lesson(lesson_id):
    conn = current_app.config["DB_CONN"]
    lesson, course = _lesson_with_course(conn, lesson_id)
    if lesson is None or course["instructor_id"] != g.user_id:
        return jsonify(error="lesson not found"), 404

    body = request.get_json(silent=True) or {}
    fields = {}
    if "title" in body:
        title = (body.get("title") or "").strip()
        if not title:
            return jsonify(error="title cannot be blank"), 400
        fields["title"] = title
    if "content" in body and lesson["content_type"] != "file":
        fields["content"] = body.get("content") or ""
    if "order_index" in body:
        fields["order_index"] = body["order_index"]

    if fields:
        try:
            with db.transaction(conn):
                db.update_lesson(conn, lesson_id, **fields)
        except sqlite3.IntegrityError:
            return jsonify(
                error=f"a lesson with order_index {fields.get('order_index')} already exists in this course"
            ), 409
    return jsonify(db.row_to_dict(db.get_lesson(conn, lesson_id)))


@bp.delete("/lessons/<int:lesson_id>")
@require_auth
@require_role("instructor")
def delete_lesson(lesson_id):
    conn = current_app.config["DB_CONN"]
    lesson, course = _lesson_with_course(conn, lesson_id)
    if lesson is None or course["instructor_id"] != g.user_id:
        return jsonify(error="lesson not found"), 404

    with db.transaction(conn):
        db.delete_lesson(conn, lesson_id)
    return jsonify(deleted=True, lesson_id=lesson_id)


@bp.post("/lessons/<int:lesson_id>/upload")
@require_auth
@require_role("instructor")
def upload_lesson_content(lesson_id):
    conn = current_app.config["DB_CONN"]
    lesson, course = _lesson_with_course(conn, lesson_id)
    if lesson is None or course["instructor_id"] != g.user_id:
        return jsonify(error="lesson not found"), 404

    if "file" not in request.files:
        return jsonify(error="no file provided (expected multipart field 'file')"), 400
    upload = request.files["file"]
    if not upload.filename:
        return jsonify(error="no file selected"), 400

    filename = secure_filename(upload.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        return jsonify(
            error=f"file type '.{ext}' not allowed; allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}"
        ), 400

    # Read into memory to enforce the size cap explicitly (in addition to
    # Flask's MAX_CONTENT_LENGTH, which rejects at the request level --
    # this catches the case where a client streams a large single file
    # under the request-level cap but the file itself is still too big).
    data = upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return jsonify(error=f"file exceeds {MAX_UPLOAD_BYTES} byte limit"), 413
    if len(data) == 0:
        return jsonify(error="uploaded file is empty"), 400

    upload_dir = current_app.config["UPLOAD_DIR"]
    os.makedirs(upload_dir, exist_ok=True)
    stored_name = f"{lesson_id}_{uuid.uuid4().hex}_{filename}"
    dest_path = os.path.join(upload_dir, stored_name)
    with open(dest_path, "wb") as f:
        f.write(data)

    with db.transaction(conn):
        db.update_lesson(conn, lesson_id, content_type="file", content=stored_name)
    return jsonify(db.row_to_dict(db.get_lesson(conn, lesson_id))), 200


@bp.post("/lessons/<int:lesson_id>/complete")
@require_auth
@require_role("student")
def complete_lesson(lesson_id):
    conn = current_app.config["DB_CONN"]
    lesson, course = _lesson_with_course(conn, lesson_id)
    if lesson is None:
        return jsonify(error="lesson not found"), 404

    enrollment = db.get_enrollment(conn, g.user_id, course["id"])
    if enrollment is None:
        return jsonify(error="not enrolled in this lesson's course"), 403

    with db.transaction(conn):
        db.mark_lesson_complete(conn, enrollment["id"], lesson_id)
    completed_ids = db.list_completed_lesson_ids(conn, enrollment["id"])
    return jsonify(lesson_id=lesson_id, completed=lesson_id in completed_ids)
