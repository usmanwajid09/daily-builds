# Forum API

A tiny REST mini-forum built with Flask. Threads and posts live in an
in-memory store (fresh on every app start — nothing is written to
disk). Create threads, post inside them, upvote posts, search threads
by keyword, and close a thread when the discussion is done.

## Why

A forum is the classic nested-resource shape: posts belong to
threads, and a thread's state (open vs closed) gates what you can do
inside it. This app shows that shape — including the rule that a
closed thread rejects new posts.

## Usage

```bash
pip install -r requirements.txt
python -m forum_api.demo
```

The demo creates two threads, adds posts with upvotes, searches for
"flask", closes a thread, and shows the closed thread rejecting a new
post.

To run the app for real:

```bash
flask --app "forum_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/threads` | Create a thread: `{"title": str, "author": str}` |
| GET | `/threads` | List threads, optionally filtered by `?q=keyword` |
| GET | `/threads/<id>` | Fetch one thread with its post count |
| POST | `/threads/<id>/close` | Close a thread |
| POST | `/threads/<id>/posts` | Add a post: `{"author": str, "content": str}` |
| GET | `/threads/<id>/posts` | List a thread's posts, most upvoted first |
| GET | `/posts/<id>` | Fetch one post |
| POST | `/posts/<id>/upvote` | Upvote a post |

A thread looks like
`{"id": 1, "title": "Best Python tips for beginners", "author":
"Usman", "status": "open", "post_count": 2}` and a post like
`{"id": 1, "thread_id": 1, "author": "Usman", "content": "Use list
comprehensions!", "upvotes": 3}`.

Validation errors return `400` with `{"error": "..."}`; unknown
threads, posts, and routes return `404`. Posting to a closed thread
or closing an already-closed thread returns `409`.

## Demo output

```
Created thread: Best Python tips for beginners (id 1)
Created thread: Flask vs Django in 2026 (id 2)
Post #1 by Usman: 3 upvotes — 'Use list comprehensions!'
Post #2 by Bilal: 1 upvotes — 'f-strings beat .format().'
Post #3 by Chen: 2 upvotes — 'Flask for small APIs.'
Search 'flask' found 1 thread(s): ['Flask vs Django in 2026']
Thread 2 is now closed
```

## Run tests

```bash
python -m pytest tests/test_forum_api.py -v
```
