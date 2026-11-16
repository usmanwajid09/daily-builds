"""Demo: submit feedback, filter by rating, and show the rating summary."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback_api.app import create_app


def main():
    client = create_app().test_client()

    for name, rating, comment in [
        ("Ayesha", 5, "Brilliant service, very fast delivery."),
        ("Bilal", 4, "Good overall, packaging could improve."),
        ("Sara", 5, "Loved it, will order again."),
        ("Omar", 2, "Late delivery and cold food."),
    ]:
        entry = client.post(
            "/feedback",
            json={"name": name, "rating": rating, "comment": comment},
        ).get_json()
        print(f"Submitted feedback: {entry}")

    print(f"Summary: {client.get('/feedback/summary').get_json()}")
    print(f"5-star feedback: {client.get('/feedback?rating=5').get_json()}")

    bad = client.post("/feedback", json={"rating": 6, "comment": "Too high"})
    print(f"Rating 6 rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
