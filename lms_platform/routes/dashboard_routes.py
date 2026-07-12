"""Role-aware summary dashboard: one endpoint, two shapes.

Students get their enrolled courses with progress/quiz/assignment/
certificate status; instructors get their owned courses with
enrollment counts, quiz pass rates, and an ungraded-submissions queue
(the thing an instructor actually needs to act on).
"""
from flask import Blueprint, current_app, g, jsonify

from .. import db
from ..decorators import require_auth
from ..progress_report import build_student_course_report

bp = Blueprint("dashboard_routes", __name__, url_prefix="/api/me")


def _student_dashboard(conn, student_id: int) -> dict:
    enrollments = db.list_enrollments_for_student(conn, student_id)
    certs_by_course = {
        c["course_id"]: c for c in db.list_certificates_for_student(conn, student_id)
    }
    courses_out = []
    for e in enrollments:
        course = db.get_course(conn, e["course_id"])
        if course is None:
            continue
        report = build_student_course_report(conn, student_id, course["id"])
        cert = certs_by_course.get(course["id"])
        courses_out.append({
            "course_id": course["id"],
            "course_title": course["title"],
            "lesson_percent_complete": report["lesson_percent_complete"],
            "quizzes_passed": sum(1 for q in report["quizzes"] if q["passed"]),
            "quizzes_total": len(report["quizzes"]),
            "assignments_submitted": sum(1 for a in report["assignments"] if a["submitted"]),
            "assignments_total": len(report["assignments"]),
            "certificate_earned": cert is not None,
            "certificate_code": cert["verification_code"] if cert else None,
        })
    return {
        "role": "student",
        "courses_enrolled": len(courses_out),
        "certificates_earned": len(certs_by_course),
        "courses": courses_out,
    }


def _instructor_dashboard(conn, instructor_id: int) -> dict:
    courses = db.list_courses(conn, instructor_id=instructor_id)
    courses_out = []
    total_ungraded = 0
    for course in courses:
        enrollments = conn.execute(
            "SELECT student_id FROM enrollments WHERE course_id = ?", (course["id"],)
        ).fetchall()
        student_count = len(enrollments)

        quizzes = db.list_quizzes_for_course(conn, course["id"])
        quiz_pass_rates = []
        for quiz in quizzes:
            attempts = db.list_attempts_for_quiz(conn, quiz["id"])
            # One attempt per student counted (their best), not every retry.
            best_by_student: dict[int, float] = {}
            passed_by_student: dict[int, bool] = {}
            for a in attempts:
                sid = a["student_id"]
                if sid not in best_by_student or a["percent"] > best_by_student[sid]:
                    best_by_student[sid] = a["percent"]
                    passed_by_student[sid] = bool(a["passed"])
            if passed_by_student:
                rate = round(100 * sum(passed_by_student.values()) / len(passed_by_student), 1)
                quiz_pass_rates.append(rate)

        assignments = db.list_assignments_for_course(conn, course["id"])
        ungraded_count = 0
        for a in assignments:
            for s in db.list_submissions_for_assignment(conn, a["id"]):
                if s["grade"] is None:
                    ungraded_count += 1
        total_ungraded += ungraded_count

        courses_out.append({
            "course_id": course["id"],
            "course_title": course["title"],
            "student_count": student_count,
            "quiz_count": len(quizzes),
            "avg_quiz_pass_rate": (
                round(sum(quiz_pass_rates) / len(quiz_pass_rates), 1) if quiz_pass_rates else None
            ),
            "assignment_count": len(assignments),
            "ungraded_submissions": ungraded_count,
        })
    return {
        "role": "instructor",
        "courses_owned": len(courses_out),
        "ungraded_submissions_total": total_ungraded,
        "courses": courses_out,
    }


@bp.get("/dashboard")
@require_auth
def dashboard():
    conn = current_app.config["DB_CONN"]
    if g.role == "student":
        return jsonify(_student_dashboard(conn, g.user_id))
    return jsonify(_instructor_dashboard(conn, g.user_id))
