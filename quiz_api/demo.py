"""Demo the quiz_api: create a quiz, submit answers, show the score."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quiz_api.app import create_app

client = create_app().test_client()

quiz = {
    "title": "Python basics",
    "questions": [
        {"question": "What does len([1, 2, 3]) return?",
         "options": ["2", "3", "4", "0"], "correct_index": 1},
        {"question": "Which keyword defines a function?",
         "options": ["def", "func", "function", "lambda"], "correct_index": 0},
        {"question": "What is 7 // 2 in Python?",
         "options": ["3.5", "3", "2", "4"], "correct_index": 1},
    ],
}
created = client.post("/quizzes", json=quiz)
print(f"Created quiz {created.get_json()['id']}: "
      f"{created.get_json()['title']} "
      f"({created.get_json()['question_count']} questions) -> "
      f"{created.status_code}")

listed = client.get("/quizzes").get_json()
print(f"Quizzes listed: {len(listed)}")

perfect = client.post("/quizzes/1/submit", json={"answers": [1, 0, 1]}).get_json()
print(f"Perfect attempt: {perfect['score']}/{perfect['total']} "
      f"({perfect['percentage']}%)")

partial = client.post("/quizzes/1/submit", json={"answers": [0, 0, 1]}).get_json()
print(f"Partial attempt: {partial['score']}/{partial['total']} "
      f"({partial['percentage']}%), per-question: "
      f"{[r['correct'] for r in partial['results']]}")

bad = client.post("/quizzes", json={"title": "Broken", "questions": []})
print(f"Empty questions -> {bad.status_code}: {bad.get_json()['error']}")
missing = client.get("/quizzes/99")
print(f"Unknown quiz -> {missing.status_code}: {missing.get_json()['error']}")
