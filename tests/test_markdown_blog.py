"""Tests for the markdown_blog Flask app, using Flask's test client."""
import pytest

from markdown_blog.app import create_app, render_markdown


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_index_lists_both_posts(client):
    html = client.get("/").get_data(as_text=True)
    assert "Hello, World" in html
    assert "Markdown, Minus the Library" in html
    assert "/post/hello-world" in html


def test_post_renders_heading_bold_list_and_link(client):
    resp = client.get("/post/hello-world")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "<h1>Hello, World</h1>" in html
    assert "<strong>tiny blog</strong>" in html
    assert "<li>Fast to read</li>" in html
    assert '<a href="https://example.com">my site</a>' in html


def test_second_post_renders_h2_h3_italic_and_code(client):
    html = client.get("/post/markdown-minus-library").get_data(as_text=True)
    assert "<h2>Why write your own renderer?</h2>" in html
    assert "<h3>What it supports</h3>" in html
    assert "<em>small</em>" in html
    assert "<code>code</code>" in html


def test_missing_post_is_404(client):
    resp = client.get("/post/no-such-post")
    assert resp.status_code == 404
    assert "Post not found" in resp.get_data(as_text=True)


def test_render_markdown_paragraphs_and_escaping():
    html = render_markdown("Hello <b>world</b>\n\nSecond para")
    assert "<p>Hello &lt;b&gt;world&lt;/b&gt;</p>" in html
    assert "<p>Second para</p>" in html
    assert "<b>world</b>" not in html


def test_render_markdown_empty_and_heading_levels():
    assert render_markdown("") == ""
    assert render_markdown("# One").startswith("<h1>One</h1>")
    assert render_markdown("## Two").startswith("<h2>Two</h2>")
    assert render_markdown("### Three").startswith("<h3>Three</h3>")


def test_render_markdown_rejects_non_string():
    with pytest.raises(TypeError):
        render_markdown(None)


def test_custom_posts_are_served():
    app = create_app(
        posts=[{"slug": "x", "title": "X", "body": "## Hi\n\n*hey*"}]
    )
    client = app.test_client()
    assert client.get("/").status_code == 200
    html = client.get("/post/x").get_data(as_text=True)
    assert "<h2>Hi</h2>" in html and "<em>hey</em>" in html
    assert client.get("/post/hello-world").status_code == 404
