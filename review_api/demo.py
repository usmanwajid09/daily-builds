"""Demo: submit reviews, vote helpful, flag one and check the summary."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from review_api.app import create_app


def main():
    client = create_app().test_client()

    client.post(
        "/reviews",
        json={"product_id": "widget-1", "author": "Alice", "rating": 5,
              "title": "Excellent", "comment": "Works perfectly."},
    )
    client.post(
        "/reviews",
        json={"product_id": "widget-1", "author": "Bob", "rating": 4,
              "title": "Good value", "comment": "Solid for the price."},
    )
    spam = client.post(
        "/reviews",
        json={"product_id": "widget-1", "author": "Spammer", "rating": 1,
              "title": "Buy now", "comment": "Visit my site!"},
    ).get_json()
    client.post("/reviews/1/helpful")
    client.post("/reviews/1/helpful")
    client.post(f"/reviews/{spam['id']}/flag")
    client.post(f"/reviews/{spam['id']}/moderate", json={"action": "hide"})

    summary = client.get("/products/widget-1/summary").get_json()
    print(
        f"widget-1: {summary['count']} reviews, "
        f"average rating {summary['average_rating']}, "
        f"distribution {summary['distribution']}"
    )
    visible = client.get("/reviews", query_string={"product_id": "widget-1"}).get_json()
    for review in visible:
        print(
            f"#{review['id']} {review['author']}: {review['rating']} stars, "
            f"{review['helpful_votes']} helpful votes, status {review['status']}"
        )
    hidden = client.get(
        "/reviews", query_string={"status": "hidden"}
    ).get_json()
    print(f"{len(hidden)} hidden review(s) after moderation")


if __name__ == "__main__":
    main()
