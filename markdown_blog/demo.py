"""Demo: render a sample post to HTML via Flask's test client."""
from markdown_blog.app import create_app


def main():
    client = create_app().test_client()

    index = client.get("/").get_data(as_text=True)
    print(f"Index page lists {index.count('/post/')} posts.")

    html = client.get("/post/hello-world").get_data(as_text=True)
    snippet = html[html.find("<p>Welcome"):]
    print("Rendered 'Hello, World' (snippet):")
    print(snippet[:300])


if __name__ == "__main__":
    main()
