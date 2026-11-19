"""Demo: create a survey, submit responses, read the per-question summary."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survey_api.app import create_app


def main():
    client = create_app().test_client()

    survey = client.post(
        "/surveys",
        json={
            "title": "Course feedback",
            "questions": [
                {
                    "text": "Which topic was most useful?",
                    "type": "choice",
                    "options": ["Databases", "AI", "Networks"],
                },
                {"text": "Rate the course overall", "type": "rating"},
                {"text": "Anything to improve?", "type": "text"},
            ],
        },
    ).get_json()
    print(f"Created survey #{survey['id']}: {survey['title']!r} "
          f"with {len(survey['questions'])} questions")

    for answers in (
        {"1": "AI", "2": 5, "3": "More labs, please."},
        {"1": "Databases", "2": 4, "3": "Faster pace."},
        {"1": "AI", "2": 5, "3": "Great course!"},
    ):
        resp = client.post(
            f"/surveys/{survey['id']}/responses", json={"answers": answers}
        )
        print(f"Submitted response #{resp.get_json()['id']} "
              f"(status {resp.status_code})")

    bad = client.post(
        f"/surveys/{survey['id']}/responses",
        json={"answers": {"1": "AI", "2": 6, "3": "Too high a rating."}},
    )
    print(f"Rating 6 rejected: {bad.status_code} {bad.get_json()}")

    incomplete = client.post(
        f"/surveys/{survey['id']}/responses",
        json={"answers": {"1": "AI", "2": 4}},
    )
    print(f"Missing answer rejected: {incomplete.status_code} "
          f"{incomplete.get_json()}")

    summary = client.get(f"/surveys/{survey['id']}/summary").get_json()
    print(f"Summary over {summary['response_count']} responses:")
    for q in summary["questions"]:
        if q["type"] == "choice":
            print(f"  Q{q['question_id']} counts: {q['counts']}")
        elif q["type"] == "rating":
            print(f"  Q{q['question_id']} average rating: {q['average']}")
        else:
            print(f"  Q{q['question_id']} text answers: {len(q['answers'])}")


if __name__ == "__main__":
    main()
