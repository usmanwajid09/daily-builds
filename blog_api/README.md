# blog_api

A small Flask blog API: posts with URL slugs derived from the title,
draft/published status, tags and search, kept in memory (a fresh store
per `create_app()` call, nothing written to disk).

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/posts` | Create a post: `{"title", "content", "tags"?, "status"?}` |
| GET | `/posts` | List posts; `?status=draft\|published`, `?tag=`, `?search=` filters |
| GET | `/posts/<slug>` | Fetch one post |
| PUT | `/posts/<slug>` | Update title, content, tags and/or status |
| POST | `/posts/<slug>/publish` | Set status to `published` |
| POST | `/posts/<slug>/unpublish` | Set status back to `draft` |
| DELETE | `/posts/<slug>` | Delete a post |

Slugs are generated from the title (`"Hello, Flask!"` →
`hello-flask`) and made unique (`hello-flask-2`, …). A slug never
changes after creation, even if the title is edited. Duplicate tags are
removed and lower-cased.

Validation: missing/empty `title` or `content`, a title with no letters
or digits, a bad `status` or bad `tags` returns JSON `400`; an unknown
slug returns JSON `404`.

## Run

```bash
python blog_api/demo.py
python -m pytest tests/test_blog_api.py -v
```
