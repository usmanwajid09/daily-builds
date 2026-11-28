"""Demo: list and filter questions, answer, add a question, show the leaderboard."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trivia_api.app import create_app


def main():
    client = create_app().test_client()

    questions = client.get("/questions").get_json()
    print(f"Seeded questions: {len(questions)}")
    print(f"First question: {questions[0]['question']} [{questions[0]['category']}/{questions[0]['difficulty']}]")

    science = client.get("/questions", query_string={"category": "science"}).get_json()
    print(f"Science questions: {len(science)}")

    new = client.post(
        "/questions",
        json={
            "question": "What is the largest ocean on Earth?",
            "options": ["Atlantic", "Indian", "Pacific", "Arctic"],
            "answer_index": 2,
            "category": "geography",
            "difficulty": "easy",
        },
    ).get_json()
    print(f"Added question #{new['id']}: {new['question']} (answer: {new['options'][new['answer_index']]})")

    correct = client.post(
        "/answer",
        json={"player": "Usman", "question_id": new["id"], "answer_index": 2},
    ).get_json()
    print(f"Usman answered correctly: {correct['correct']} (score {correct['score']}/{correct['answered']})")

    wrong = client.post(
        "/answer",
        json={"player": "Usman", "question_id": new["id"], "answer_index": 0},
    ).get_json()
    print(f"Usman answered correctly: {wrong['correct']} (score {wrong['score']}/{wrong['answered']})")

    board = client.get("/scores").get_json()
    print(f"Leaderboard: {[(row['player'], row['score'], row['answered'], row['accuracy']) for row in board]}")


if __name__ == "__main__":
    main()
