"""Certificate-eligibility rules, kept as pure functions over DB rows
so the eligibility policy is unit-testable without spinning up Flask.

Policy (documented here since it's a judgment call, not a spec
handed down anywhere): a student earns a course certificate once
they've (1) completed every lesson in the course, (2) passed every
quiz in the course (their best attempt's percent >= that quiz's
passing_score), and (3) submitted something for every assignment in
the course. Submission, not a passing grade, is the assignment bar --
grading is manual and instructor-paced, so gating a certificate on a
grade that may not exist yet would make certificates hostage to
instructor turnaround time. A course with zero lessons/quizzes/
assignments trivially satisfies that component (nothing to complete).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reasons: list[str] = field(default_factory=list)


def check_certificate_eligibility(
    *,
    total_lessons: int,
    completed_lessons: int,
    quizzes: list[dict],  # [{"id":.., "title":.., "passed": bool}, ...]
    assignments: list[dict],  # [{"id":.., "title":.., "submitted": bool}, ...]
) -> EligibilityResult:
    reasons = []
    if completed_lessons < total_lessons:
        reasons.append(
            f"{total_lessons - completed_lessons} of {total_lessons} lesson(s) not yet completed"
        )
    for q in quizzes:
        if not q["passed"]:
            reasons.append(f"quiz '{q['title']}' not yet passed")
    for a in assignments:
        if not a["submitted"]:
            reasons.append(f"assignment '{a['title']}' not yet submitted")
    return EligibilityResult(eligible=not reasons, reasons=reasons)
