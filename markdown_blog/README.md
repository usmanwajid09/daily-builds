# Markdown Blog

A tiny blog built with Flask, whose Markdown renderer is written from
scratch — no `markdown` library, just the standard library. Posts live
in memory, seeded with two samples.

## Why

A blog is the classic first Flask app, and writing the renderer by hand
makes the parsing visible: block structure (headings, lists,
paragraphs) on one pass, inline formatting (bold, italic, code, links)
on another, with raw HTML escaped so a post can never inject markup.

## Usage

```bash
pip install -r requirements.txt
python -m markdown_blog.demo
```

The demo fetches the index and renders the "Hello, World" post through
Flask's test client, printing a snippet of the HTML. To run the app for
real:

```bash
flask --app "markdown_blog.app:create_app" run
```

## Markdown subset supported

- `#`, `##`, `###` headings
- `**bold**` and `*italic*`
- `` `inline code` ``
- `[text](https://url)` links
- `-` bullet lists
- Paragraphs (blank-line separated)

## Routes

| Route | Description |
| ----- | ----------- |
| `GET /` | List of posts |
| `GET /post/<slug>` | One rendered post, or a 404 page |

## Run tests

```bash
python -m pytest tests/test_markdown_blog.py -v
```
