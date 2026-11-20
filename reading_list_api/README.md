# reading_list_api

A small Flask reading-list API: books with title, author, total pages and
pages read, kept in memory (a fresh store per `create_app()` call, nothing
written to disk). Reading progress is tracked as a percentage and a book is
marked `finished` automatically when the last page is logged.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/books` | Create a book: `{"title", "author", "total_pages", "status"?, "pages_read"?}` |
| GET | `/books` | List books; `?status=to-read\|reading\|finished`, `?author=` (substring, case-insensitive) |
| GET | `/books/<id>` | Fetch one book |
| PUT | `/books/<id>` | Update title, author, status, total_pages and/or pages_read |
| DELETE | `/books/<id>` | Delete a book |
| POST | `/books/<id>/read` | Log pages read: `{"pages": int}`; caps at total_pages, auto-finishes |
| POST | `/books/<id>/finish` | Mark a book finished (pages_read = total_pages) |
| GET | `/stats` | Totals, counts by status and overall progress % |

Each book is `{"id", "title", "author", "status", "total_pages",
"pages_read", "progress_pct"}`. `progress_pct` is
`pages_read / total_pages * 100`, rounded to one decimal.

Validation: missing/empty `title` or `author`, a non-positive
`total_pages`, `pages_read` above `total_pages`, a bad `status`, or a
`finished` book whose pages read do not equal the total all return JSON
`400`; an unknown id returns JSON `404`.

## Run

```bash
python reading_list_api/demo.py
python -m pytest tests/test_reading_list_api.py -v
```

## Examples

```bash
curl -X POST localhost:5000/books -H 'Content-Type: application/json' \
  -d '{"title": "Dune", "author": "Frank Herbert", "total_pages": 688}'

curl 'localhost:5000/books?author=herbert'
curl -X POST localhost:5000/books/1/read -H 'Content-Type: application/json' \
  -d '{"pages": 50}'
curl localhost:5000/stats
```
