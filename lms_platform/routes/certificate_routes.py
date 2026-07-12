"""Certificate issuance (auto-graded eligibility check) and public
verification. Verification is deliberately unauthenticated -- a
certificate is meant to be checkable by anyone holding the code (an
employer, e.g.), the same way real credential-verification pages work.
"""
import uuid

from flask import Blueprint, current_app, g, jsonify

from .. import db
from ..certificates import check_certificate_eligibility
from ..decorators import require_auth, require_role
from ..progress_report import build_student_course_report

bp = Blueprint("certificate_routes", __name__, url_prefix="/api")


@bp.post("/courses/<int:course_id>/certificate")
@require_auth
@require_role("student")
def issue_certificate(course_id):
    conn = current_app.config["DB_CONN"]
    course = db.get_course(conn, course_id)
    if course is None:
        return jsonify(error="course not found"), 404

    enrollment = db.get_enrollment(conn, g.user_id, course_id)
    if enrollment is None:
        return jsonify(error="not enrolled in this course"), 403

    existing = db.get_certificate_for_course(conn, g.user_id, course_id)
    if existing is not None:
        return jsonify(db.row_to_dict(existing)), 200

    report = build_student_course_report(conn, g.user_id, course_id)
    result = check_certificate_eligibility(
        total_lessons=report["total_lessons"],
        completed_lessons=report["completed_lessons"],
        quizzes=[{"title": q["title"], "passed": q["passed"]} for q in report["quizzes"]],
        assignments=[{"title": a["title"], "submitted": a["submitted"]}
                     for a in report["assignments"]],
    )
    if not result.eligible:
        return jsonify(error="not yet eligible for a certificate", reasons=result.reasons), 403

    code = uuid.uuid4().hex
    with db.transaction(conn):
        db.create_certificate(conn, g.user_id, course_id, code)
    return jsonify(db.row_to_dict(db.get_certificate_for_course(conn, g.user_id, course_id))), 201


@bp.get("/certificates/<code>")
def verify_certificate(code):
    """Public: no auth required. Returns 404 (not 400) for a malformed
    or unknown code alike, so this can't be used to probe which codes
    are syntactically valid vs. simply don't exist."""
    conn = current_app.config["DB_CONN"]
    cert = db.get_certificate_by_code(conn, code)
    if cert is None:
        return jsonify(error="certificate not found"), 404
    student = db.get_user(conn, cert["student_id"])
    course = db.get_course(conn, cert["course_id"])
    return jsonify(
        verification_code=cert["verification_code"],
        student_email=student["email"] if student else None,
        course_title=course["title"] if course else None,
        issued_at=cert["issued_at"],
    )


@bp.get("/me/certificates")
@require_auth
@require_role("student")
def my_certificates():
    conn = current_app.config["DB_CONN"]
    certs = db.list_certificates_for_student(conn, g.user_id)
    out = []
    for c in certs:
        course = db.get_course(conn, c["course_id"])
        out.append({
            "verification_code": c["verification_code"],
            "course_id": c["course_id"],
            "course_title": course["title"] if course else None,
            "issued_at": c["issued_at"],
        })
    return jsonify(certificates=out)
