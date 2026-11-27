"""Demo: add articles, search them, find related ones."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base_api.app import create_app


def main():
    client = create_app().test_client()

    articles = []
    for title, body, category, tags in (
        (
            "Getting started with Flask",
            "Flask is a lightweight Python web framework. Install it "
            "with pip and create your first app in minutes.",
            "Guides",
            ["flask", "python", "web"],
        ),
        (
            "Flask testing with pytest",
            "Use Flask's test client to exercise your routes. The "
            "Flask test client makes it easy to check status codes.",
            "Guides",
            ["flask", "testing", "python"],
        ),
        (
            "Deploying a Python API",
            "Ship your Python API with gunicorn behind nginx, and "
            "keep secrets out of the repository.",
            "Deployment",
            ["python", "deployment"],
        ),
    ):
        article = client.post(
            "/articles",
            json={"title": title, "body": body, "category": category,
                  "tags": tags},
        ).get_json()
        articles.append(article)
        print(f"Created article #{article['id']}: {article['title']!r} "
              f"({article['category']}, tags {article['tags']})")

    results = client.get("/search?q=flask").get_json()
    print(f"Search 'flask': {len(results)} result(s)")
    for result in results:
        print(f"  #{result['id']} {result['title']!r} "
              f"(score {result['score']}): {result['snippet']}")

    related = client.get(
        f"/articles/{articles[0]['id']}/related"
    ).get_json()
    print(f"Related to {articles[0]['title']!r}: {len(related)} article(s)")
    for item in related:
        print(f"  #{item['id']} {item['title']!r} "
              f"(shared tags {item['shared_tags']})")

    categories = client.get("/categories").get_json()
    print("Categories: " + ", ".join(
        f"{c['category']} ({c['article_count']})" for c in categories
    ))


if __name__ == "__main__":
    main()
