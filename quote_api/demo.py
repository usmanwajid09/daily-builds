"""Demo: add quotes, filter by author, and fetch a random one."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quote_api.app import create_app


def main():
    client = create_app().test_client()

    first = client.post(
        "/quotes",
        json={
            "text": "Simplicity is the soul of efficiency.",
            "author": "Austin Freeman",
            "tags": ["simplicity"],
        },
    ).get_json()
    print(f"Added quote: {first}")
    client.post(
        "/quotes",
        json={
            "text": "Talk is cheap. Show me the code.",
            "author": "Linus Torvalds",
            "tags": ["code", "programming"],
        },
    )

    by_author = client.get("/quotes?author=torvalds").get_json()
    print(f"Quotes by Torvalds: {by_author}")
    tagged = client.get("/quotes?tag=code").get_json()
    print(f"Quotes tagged 'code': {tagged}")

    random_quote = client.get("/quotes/random").get_json()
    print(f"Random quote: {random_quote}")

    all_quotes = client.get("/quotes").get_json()
    print(f"All quotes ({len(all_quotes)}): {all_quotes}")


if __name__ == "__main__":
    main()
