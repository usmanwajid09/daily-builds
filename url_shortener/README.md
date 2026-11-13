# URL Shortener

A tiny URL shortener built with Flask. Links live in an in-memory store
(fresh on every app start), and short codes are base-62 encodings of an
incrementing counter, so they are unique by construction.

## Why

Shortening a URL touches the whole request cycle — validating input,
generating an identifier, redirecting, and keeping per-link stats — in
about a hundred lines.

## Usage

```bash
pip install -r requirements.txt
python -m url_shortener.demo
```

The demo shortens a URL, follows the redirect, and prints the click
stats. To run the app for real:

```bash
flask --app "url_shortener.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/shorten` | `{"url": "https://..."}` → `201` with `code` and `short_url` |
| GET | `/<code>` | `302` redirect to the original URL (counts a click) |
| GET | `/stats/<code>` | `{"code", "url", "clicks"}` |

Only `http://` and `https://` URLs with a host are accepted; anything
else returns `400` with `{"error": "..."}`. Unknown codes return `404`.

## Run tests

```bash
python -m pytest tests/test_url_shortener.py -v
```
