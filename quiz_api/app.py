"""Quiz API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /quizzes`` — create a quiz
  (``{"title": str, "questions": [{"question": str, "options": [str, ...],
  "correct_index": int}]}``). Each question needs at least 2 options
  and a ``correct_index`` inside the options.
- ``GET /quizzes`` — list quizzes (titles, question counts, no answers)
- ``GET /quizzes/<id>`` — fetch one quiz (without revealing answers)
- ``POST /quizzes/<id>/submit`` — submit answers
  (``{"answers": [int, ...]}``) and get a score, percentage and a
  per-question breakdown
- ``DELETE /quizzes/<id>`` — delete a quiz

Bad payloads return 400 with a clear message; unknown quiz ids 404.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory quiz store."""
    app = Flask(__name__)
    app.config["QUIZZES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _validate_questions(questions):
        if not isinstance(questions, list) or not questions:
            return "questions must be a non-empty list"
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                return f"question {i} must be an object"
            if not isinstance(q.get("question"), str) or not q["question"].strip():
                return f"question {i}: 'question' must be a non-empty string"
            options = q.get("options")
            if not isinstance(options, list) or len(options) < 2:
                return f"question {i}: 'options' must be a list of at least 2 strings"
            if not all(isinstance(o, str) and o.strip() for o in options):
                return f"question {i}: every option must be a non-empty string"
            ci = q.get("correct_index")
            if not isinstance(ci, int) or isinstance(ci, bool):
                return f"question {i}: 'correct_index' must be an integer"
            if not 0 <= ci < len(options):
                return f"question {i}: 'correct_index' is out of range"
        return None

    def _public(quiz):
        return {
            "id": quiz["id"],
            "title": quiz["title"],
            "question_count": len(quiz["questions"]),
            "questions": [
                {"question": q["question"], "options": q["options"]}
                for q in quiz["questions"]
            ],
        }

    def _get_quiz(quiz_id):
        quiz = app.config["QUIZZES"].get(quiz_id)
        if quiz is None:
            return None, _error(f"quiz {quiz_id} not found", 404)
        return quiz, None

    @app.post("/quizzes")
    def create_quiz():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        problem = _validate_questions(data.get("questions"))
        if problem:
            return _error(problem, 400)
        quiz_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        quiz = {
            "id": quiz_id,
            "title": title.strip(),
            "questions": [
                {
                    "question": q["question"].strip(),
                    "options": list(q["options"]),
                    "correct_index": q["correct_index"],
                }
                for q in data["questions"]
            ],
        }
        app.config["QUIZZES"][quiz_id] = quiz
        return jsonify(_public(quiz)), 201

    @app.get("/quizzes")
    def list_quizzes():
        quizzes = sorted(app.config["QUIZZES"].values(), key=lambda q: q["id"])
        return jsonify([_public(q) for q in quizzes])

    @app.get("/quizzes/<int:quiz_id>")
    def get_quiz(quiz_id):
        quiz, err = _get_quiz(quiz_id)
        if err is not None:
            return err
        return jsonify(_public(quiz))

    @app.post("/quizzes/<int:quiz_id>/submit")
    def submit_quiz(quiz_id):
        quiz, err = _get_quiz(quiz_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        answers = data.get("answers")
        if not isinstance(answers, list):
            return _error("answers must be a list of option indexes", 400)
        if len(answers) != len(quiz["questions"]):
            return _error(
                f"answers must have {len(quiz['questions'])} entries, "
                f"got {len(answers)}",
                400,
            )
        results = []
        score = 0
        for i, (q, ans) in enumerate(zip(quiz["questions"], answers)):
            if not isinstance(ans, int) or isinstance(ans, bool):
                return _error(f"answer {i} must be an integer option index", 400)
            if not 0 <= ans < len(q["options"]):
                return _error(f"answer {i} is out of range", 400)
            correct = ans == q["correct_index"]
            score += int(correct)
            results.append(
                {
                    "question": i,
                    "correct": correct,
                    "correct_index": q["correct_index"],
                }
            )
        total = len(quiz["questions"])
        return jsonify(
            {
                "quiz_id": quiz_id,
                "score": score,
                "total": total,
                "percentage": round(100 * score / total, 1),
                "results": results,
            }
        )

    @app.delete("/quizzes/<int:quiz_id>")
    def delete_quiz(quiz_id):
        quiz, err = _get_quiz(quiz_id)
        if err is not None:
            return err
        del app.config["QUIZZES"][quiz_id]
        return jsonify({"deleted": quiz_id})

    return app
