# RSS Reader API

A tiny RSS-reader API built with Flask. Feeds are submitted as raw
RSS XML strings and parsed with the standard library's
`xml.etree.ElementTree` — no extra dependency. Parsed feeds live in
an in-memory store (fresh on every app start — nothing is written to
disk).

## Usage

```bash
pip install -r requirements.txt
python -m rss_reader_api.demo
```

The demo submits a two-item RSS feed, lists the feeds, prints each
item's title, date and link, and shows malformed XML being rejected
with a 400.

To run the app for real:

```bash
flask --app "rss_reader_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/feeds` | List feeds: id, name, channel title, item count |
| POST | `/feeds` | Add a feed: `{"name": str, "xml": str}` (raw RSS 2.0 XML) |
| GET | `/feeds/<id>` | One feed's details |
| GET | `/feeds/<id>/items` | The feed's items |
| DELETE | `/feeds/<id>` | Remove a feed |

Invalid XML, a non-`<rss>` root, or a missing `<channel>` all return
`400` with `{"error": "..."}`; unknown ids return `404`. An item
looks like `{"title": "...", "link": "...", "description": "...",
"pubDate": "..."}` (missing fields come back as empty strings).

## Run tests

```bash
python -m pytest tests/test_rss_reader_api.py -v
```
