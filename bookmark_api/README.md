# bookmark_api

A small Flask REST API for saving bookmarks, with URL validation, tags,
and search by tag or title. All state lives in memory; nothing is
written to disk.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/bookmarks` | List bookmarks. Optional `?tag=<tag>` and `?q=<text>` (title search). |
| POST | `/bookmarks` | Create a bookmark: `{"title": str, "url": str, "tags": [str]?}` |
| GET | `/bookmarks/<id>` | Fetch one bookmark. |
| PUT | `/bookmarks/<id>` | Update `title`, `url` and/or `tags`. |
| DELETE | `/bookmarks/<id>` | Delete a bookmark. |

URLs must be valid `http(s)` URLs with a host; tags must be a list of
non-empty strings. Bad input returns `400` with a JSON `{"error": ...}`
message; unknown ids return `404`.

## Run

```bash
python bookmark_api/demo.py
python -m pytest tests/test_bookmark_api.py -q
```

## Example

```
Created bookmark: {'id': 1, 'title': 'Flask docs', 'url': 'https://flask.palletsprojects.com/', 'tags': ['python', 'web']}
Bookmarks tagged 'python': 2
Search 'flask': [{'id': 1, 'title': 'Flask docs', ...}]
```
