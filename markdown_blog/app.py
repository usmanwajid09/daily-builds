"""Tiny blog built with Flask, with a hand-written Markdown renderer.

Posts live in memory (seeded with two sample posts). The renderer in
:func:`render_markdown` supports a small Markdown subset — headings
(``#``/``##``/``###``), ``**bold**``, ``*italic*``, ```code```,
``[text](url)`` links, and ``-`` bullet lists — and is written from
scratch with the standard library only; no markdown package is used.

Routes:

- ``GET /`` — list of posts
- ``GET /post/<slug>`` — one rendered post, or a 404 page
"""
import html
import re

from flask import Flask

POSTS = [
    {
        "slug": "hello-world",
        "title": "Hello, World",
        "body": (
            "# Hello, World\n"
            "\n"
            "Welcome to my **tiny blog**. It renders Markdown with a "
            "hand-written parser — no libraries involved.\n"
            "\n"
            "- Fast to read\n"
            "- Easy to write\n"
            "- Fun to build\n"
            "\n"
            "Read more on [my site](https://example.com)."
        ),
    },
    {
        "slug": "markdown-minus-library",
        "title": "Markdown, Minus the Library",
        "body": (
            "## Why write your own renderer?\n"
            "\n"
            "Because the subset a small blog needs is *small*: headings, "
            "emphasis, `code`, links, and lists.\n"
            "\n"
            "### What it supports\n"
            "\n"
            "- `#`, `##`, and `###` headings\n"
            "- **bold** and *italic* text\n"
            "- `inline code` spans\n"
            "- [links](https://example.com)\n"
        ),
    },
]


def _inline(text):
    """Apply inline Markdown formatting to one already-escaped line."""
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def render_markdown(source):
    """Render a small Markdown subset to an HTML fragment.

    Supports ``#``/``##``/``###`` headings, ``**bold**``, ``*italic*``,
    ```code```, ``[text](url)`` links, ``-`` bullet lists, and
    paragraphs. Raw HTML in the source is escaped, never rendered.
    """
    if not isinstance(source, str):
        raise TypeError("markdown source must be a string")
    lines = html.escape(source).splitlines()
    blocks = []
    current_list = []
    current_para = []

    def flush_para():
        if current_para:
            blocks.append("<p>" + _inline(" ".join(current_para)) + "</p>")
            current_para.clear()

    def flush_list():
        if current_list:
            items = "".join(f"<li>{_inline(item)}</li>" for item in current_list)
            blocks.append(f"<ul>{items}</ul>")
            current_list.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_para()
            flush_list()
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            flush_para()
            flush_list()
            level = len(heading.group(1))
            blocks.append(
                f"<h{level}>{_inline(heading.group(2).strip())}</h{level}>"
            )
            continue
        if stripped.startswith("- "):
            flush_para()
            current_list.append(stripped[2:].strip())
            continue
        flush_list()
        current_para.append(stripped)
    flush_para()
    flush_list()
    return "\n".join(blocks)


def _page(title, body):
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1>{html.escape(title)}</h1>{body}"
        "</body></html>"
    )


def create_app(posts=None):
    """Return a Flask app serving the given posts (or the seeded ones)."""
    app = Flask(__name__)
    app.config["POSTS"] = {p["slug"]: p for p in (posts or POSTS)}

    @app.get("/")
    def index():
        items = "".join(
            f"<li><a href=\"/post/{slug}\">{html.escape(post['title'])}</a></li>"
            for slug, post in app.config["POSTS"].items()
        )
        return _page("Tiny Blog", f"<ul>{items}</ul>")

    @app.get("/post/<slug>")
    def show_post(slug):
        post = app.config["POSTS"].get(slug)
        if post is None:
            return _page("Post not found", "<p>No post with that address.</p>"), 404
        return _page(post["title"], render_markdown(post["body"]))

    return app
