"""Shared student-progress aggregation used by both the certificate
route (eligibility check) and the dashboard route (summary view), so
the two can't drift out of sync on what "done with this course" means.
"""
from __future__ import annotations

from . import db


def build_student_course_report(conn, student_id: int, course_id: int) -> dict:
    """Everything needed to judge one student's standing in one course:
    lesson completion, per-quiz pass/fail (best attempt), and per-
    assignment submitted/graded status. Assumes the student is
    enrolled -- callers check that first."""
    enrollment = db.get_enrollment(conn, student_id, course_id)
    lessons = db.list_lessons_for_course(conn, course_id)
    completed_ids = db.list_completed_lesson_ids(conn, enrollment["id"]) if enrollment else set()
    total_lessons = len(lessons)
    completed_lessons = sum(1 for l in lessons if l["id"] in completed_ids)
    lesson_percent = round(100 * completed_lessons / total_lessons, 1) if total_lessons else 100.0

    quizzes_out = []
    for quiz in db.list_quizzes_for_course(conn, course_id):
        best = db.best_attempt_for_student(conn, student_id, quiz["id"])
        quizzes_out.append({
            "id": quiz["id"], "title": quiz["title"],
            "attempted": best is not None,
            "best_percent": best["percent"] if best else None,
            "passed": bool(best["passed"]) if best else False,
        })

    assignments_out = []
    for assignment in db.list_assignments_for_course(conn, course_id):
        submission = db.get_submission(conn, assignment["id"], student_id)
        assignments_out.append({
            "id": assignment["id"], "title": assignment["title"],
            "submitted": submission is not None,
            "grade": submission["grade"] if submission else None,
            "graded": submission is not None and submission["grade"] is not None,
        })

    return {
        "course_id": course_id,
        "total_lessons": total_lessons,
        "completed_lessons": completed_lessons,
        "lesson_percent_complete": lesson_percent,
        "quizzes": quizzes_out,
        "assignments": assignments_out,
    }
