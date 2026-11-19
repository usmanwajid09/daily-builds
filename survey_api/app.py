"""Flask survey API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /surveys`` — create a survey (``{"title": str, "questions":
  [{"text": str, "type": "choice" | "rating" | "text",
  "options": [str]?}]}``). ``choice`` questions need at least two
  options; ``rating`` questions are answered 1-5.
- ``GET /surveys`` — list surveys
- ``GET /surveys/<id>`` — fetch one survey (includes
  ``response_count``)
- ``DELETE /surveys/<id>`` — delete a survey and its responses
- ``POST /surveys/<id>/responses`` — submit a response
  (``{"answers": {"<question_id>": value}}``). Every question must be
  answered; choice answers must be one of the options, rating answers
  an integer 1-5, text answers non-empty strings.
- ``GET /surveys/<id>/responses`` — list submitted responses
- ``GET /surveys/<id>/summary`` — per-question summary: choice
  questions get a count per option, rating questions get count and
  average, text questions get the list of answers.

A survey is ``{"id": int, "title": str, "questions": [{"id": int,
"text": str, "type": str, "options": [str]}]}``.
"""
from flask import Flask, jsonify, request

QUESTION_TYPES = ("choice", "rating", "text")


def create_app():
    """Return a Flask app with a fresh, empty in-memory survey store."""
    app = Flask(__name__)
    app.config["SURVEYS"] = {}
    app.config["RESPONSES"] = {}
    app.config["NEXT_SURVEY_ID"] = 1
    app.config["NEXT_RESPONSE_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_survey(survey_id):
        survey = app.config["SURVEYS"].get(survey_id)
        if survey is None:
            return None, _error(f"survey {survey_id} not found", 404)
        return survey, None

    def _survey_view(survey):
        count = sum(
            1 for r in app.config["RESPONSES"].values()
            if r["survey_id"] == survey["id"]
        )
        return {**survey, "response_count": count}

    def _validate_questions(questions):
        if not isinstance(questions, list) or not questions:
            return "questions must be a non-empty list"
        for q in questions:
            if not isinstance(q, dict):
                return "each question must be an object"
            if not isinstance(q.get("text"), str) or not q["text"].strip():
                return "each question needs a non-empty text"
            if q.get("type") not in QUESTION_TYPES:
                return (
                    "each question type must be one of "
                    + ", ".join(QUESTION_TYPES)
                )
            if q["type"] == "choice":
                options = q.get("options")
                if not isinstance(options, list) or len(options) < 2:
                    return "choice questions need at least two options"
                for opt in options:
                    if not isinstance(opt, str) or not opt.strip():
                        return "each option must be a non-empty string"
        return None

    @app.post("/surveys")
    def create_survey():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        q_error = _validate_questions(data.get("questions"))
        if q_error:
            return _error(q_error, 400)
        survey_id = app.config["NEXT_SURVEY_ID"]
        app.config["NEXT_SURVEY_ID"] += 1
        questions = []
        for i, q in enumerate(data["questions"], start=1):
            questions.append({
                "id": i,
                "text": q["text"].strip(),
                "type": q["type"],
                "options": [o.strip() for o in q.get("options", [])]
                if q["type"] == "choice" else [],
            })
        survey = {"id": survey_id, "title": title.strip(), "questions": questions}
        app.config["SURVEYS"][survey_id] = survey
        return jsonify(_survey_view(survey)), 201

    @app.get("/surveys")
    def list_surveys():
        surveys = sorted(app.config["SURVEYS"].values(), key=lambda s: s["id"])
        return jsonify([_survey_view(s) for s in surveys])

    @app.get("/surveys/<int:survey_id>")
    def get_survey(survey_id):
        survey, err = _get_survey(survey_id)
        if err is not None:
            return err
        return jsonify(_survey_view(survey))

    @app.delete("/surveys/<int:survey_id>")
    def delete_survey(survey_id):
        _survey, err = _get_survey(survey_id)
        if err is not None:
            return err
        del app.config["SURVEYS"][survey_id]
        doomed = [
            rid for rid, r in app.config["RESPONSES"].items()
            if r["survey_id"] == survey_id
        ]
        for rid in doomed:
            del app.config["RESPONSES"][rid]
        return jsonify({"deleted": survey_id, "responses_deleted": len(doomed)})

    @app.post("/surveys/<int:survey_id>/responses")
    def submit_response(survey_id):
        survey, err = _get_survey(survey_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            return _error("answers must be an object keyed by question id", 400)
        by_id = {q["id"]: q for q in survey["questions"]}
        for qid in answers:
            try:
                key = int(qid)
            except (TypeError, ValueError):
                return _error(f"unknown question id {qid!r}", 400)
            if key not in by_id:
                return _error(f"unknown question id {qid!r}", 400)
        for q in survey["questions"]:
            key = str(q["id"])
            if key not in answers:
                return _error(f"missing answer for question {q['id']}", 400)
            value = answers[key]
            if q["type"] == "choice":
                if not isinstance(value, str) or value not in q["options"]:
                    return _error(
                        f"question {q['id']} must be one of "
                        f"{', '.join(q['options'])}",
                        400,
                    )
            elif q["type"] == "rating":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 1 <= value <= 5
                ):
                    return _error(
                        f"question {q['id']} must be an integer from 1 to 5", 400
                    )
            else:  # text
                if not isinstance(value, str) or not value.strip():
                    return _error(
                        f"question {q['id']} must be a non-empty string", 400
                    )
        response_id = app.config["NEXT_RESPONSE_ID"]
        app.config["NEXT_RESPONSE_ID"] += 1
        response = {
            "id": response_id,
            "survey_id": survey_id,
            "answers": {str(k): v for k, v in answers.items()},
        }
        app.config["RESPONSES"][response_id] = response
        return jsonify(response), 201

    @app.get("/surveys/<int:survey_id>/responses")
    def list_responses(survey_id):
        _survey, err = _get_survey(survey_id)
        if err is not None:
            return err
        responses = [
            r for r in sorted(app.config["RESPONSES"].values(), key=lambda r: r["id"])
            if r["survey_id"] == survey_id
        ]
        return jsonify(responses)

    @app.get("/surveys/<int:survey_id>/summary")
    def get_summary(survey_id):
        survey, err = _get_survey(survey_id)
        if err is not None:
            return err
        responses = [
            r for r in app.config["RESPONSES"].values()
            if r["survey_id"] == survey_id
        ]
        summary = []
        for q in survey["questions"]:
            values = [r["answers"][str(q["id"])] for r in responses]
            entry = {
                "question_id": q["id"],
                "text": q["text"],
                "type": q["type"],
                "response_count": len(values),
            }
            if q["type"] == "choice":
                entry["counts"] = {opt: values.count(opt) for opt in q["options"]}
            elif q["type"] == "rating":
                entry["average"] = (
                    round(sum(values) / len(values), 2) if values else None
                )
            else:
                entry["answers"] = values
            summary.append(entry)
        return jsonify(
            {
                "survey_id": survey_id,
                "response_count": len(responses),
                "questions": summary,
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
