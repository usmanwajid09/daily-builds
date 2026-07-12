"""Pure quiz-grading logic, kept free of Flask/DB so it's independently
unit-testable. Routes pass in plain dicts/lists built from DB rows and
get back a GradingResult; nothing here touches sqlite3 or Flask.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QuestionResult:
    question_id: int
    points: float
    awarded: float
    correct: bool
    selected_option_id: int | None


@dataclass(frozen=True)
class GradingResult:
    score: float
    total_points: float
    percent: float
    passed: bool
    question_results: list[QuestionResult] = field(default_factory=list)


def grade_quiz(questions: list[dict], correct_option_by_question: dict[int, int],
                answers_by_question: dict[int, int], passing_score: float) -> GradingResult:
    """
    questions: [{"id": int, "points": float}, ...] -- every question in the quiz.
    correct_option_by_question: {question_id: correct_option_id}.
    answers_by_question: {question_id: selected_option_id} -- a question
        missing from this dict is treated as unanswered (0 points, not
        an error), since a blank answer is a valid (if losing) submission.
    passing_score: percent (0-100) required to pass.
    """
    total_points = sum(q["points"] for q in questions)
    score = 0.0
    results = []
    for q in questions:
        qid = q["id"]
        points = q["points"]
        selected = answers_by_question.get(qid)
        correct_option = correct_option_by_question.get(qid)
        is_correct = selected is not None and selected == correct_option
        awarded = points if is_correct else 0.0
        score += awarded
        results.append(QuestionResult(
            question_id=qid, points=points, awarded=awarded,
            correct=is_correct, selected_option_id=selected,
        ))
    percent = round(100 * score / total_points, 2) if total_points else 0.0
    passed = percent >= passing_score
    return GradingResult(score=score, total_points=total_points, percent=percent,
                          passed=passed, question_results=results)
