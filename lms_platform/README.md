# lms-platform

An LMS (learning management system) backend: courses, lessons, content
upload, enrollment, and progress tracking. Part of the
`daily-projects` arc series in this repo -- see `ARC_QUEUE.md` /
`CURRENT_ARC.md` at the repo root for the overall plan.

## Milestone 1 (this milestone): data model, CRUD, upload, progress

- **Data model** (SQLite via stdlib `sqlite3`, no ORM): `users`
  (instructor or student role), `courses` (owned by an instructor),
  `lessons` (belong to a course, ordered by `order_index`),
  `enrollments` (a student joining a course), `progress` (a completed
  lesson for a specific enrollment). All child rows cascade-delete via
  `ON DELETE CASCADE` foreign keys -- deleting a course takes its
  lessons, enrollments, and progress rows with it in one statement.
- **Auth**: bcrypt password hashing + HS256 JWTs, same pattern used by
  `dev_collab_platform` and `saas_starter` elsewhere in this repo. The
  JWT `sub` claim is always encoded as a string and cast back to `int`
  on decode -- see `dev_collab_platform`'s 2026-07-11 hotfix (PR #11)
  for why: newer PyJWT enforces RFC 7519's requirement that `sub` be a
  string.
- **Course CRUD**: instructors create/update/delete their own courses
  (`POST/PATCH/DELETE /api/courses/<id>`); everyone (any authenticated
  role) can browse the full catalogue (`GET /api/courses`); an
  instructor can filter to just their own with `?mine=1`.
- **Lesson CRUD**: scoped under a course
  (`POST /api/courses/<id>/lessons`, `GET/PATCH/DELETE
  /api/lessons/<id>`), owner-only writes. A lesson's `content_type` is
  `text` (content is the body), `video_url` (content is a URL), or
  `file` -- but `file` can only be set via the upload endpoint below,
  not by claiming an arbitrary path through create/update.
- **Content upload**: `POST /api/lessons/<id>/upload` accepts a
  multipart file, validated against an extension allowlist (pdf, png,
  jpg/jpeg, gif, mp4, mp3, wav, zip, txt, md, docx, pptx, csv), a
  10&nbsp;MB per-file cap, and a rejection of empty files. Stored under
  `lms_platform/data/uploads/` with a UUID-prefixed filename
  (`secure_filename()`-sanitized) so two lessons can't collide.
- **Enrollment + progress**: a student self-enrolls
  (`POST /api/courses/<id>/enroll`, idempotent -- calling it twice
  returns the existing enrollment rather than erroring), marks lessons
  complete (`POST /api/lessons/<id>/complete`, also idempotent), and
  reads progress either per-course (`GET /api/courses/<id>/progress`)
  or across everything they're enrolled in
  (`GET /api/me/enrollments`), each returning a `percent_complete`
  computed from completed vs. total lessons (0% for a course with zero
  lessons, not a division-by-zero crash).

## Why Flask (and not the dependency-free stdlib WSGI style used by
`football_stats_site`)

Earlier arcs in this repo alternated between a raw stdlib `wsgiref`
app (`football_stats_site`) and Flask (`dev_collab_platform`,
`saas_starter`) depending on how much routing complexity was needed.
This milestone has enough nested path parameters and blueprint-shaped
concerns (auth, courses, lessons, enrollments) that Flask's routing
and `request.files` multipart handling meaningfully cut boilerplate
versus hand-rolling a regex router and a multipart parser -- so it
follows the `dev_collab_platform`/`saas_starter` precedent rather than
the `football_stats_site` one. `requirements.txt` now pins
`Flask`/`bcrypt`/`PyJWT`, which were previously implicit (already used
by earlier arcs) but never actually listed.

## Running it

```bash
cd lms_platform
python3 run.py          # dev server on http://127.0.0.1:5050 (Flask dev server, not production WSGI)
```

Or in tests / a script, use the app factory directly:

```python
from lms_platform import create_app
app = create_app(db_path=":memory:", jwt_secret="dev-secret-change-me")
client = app.test_client()
```

## Running the tests

```bash
cd lms_platform
python3 -m pytest tests -q
```

50 tests, all passing.

## API summary

| Method | Path                             | Auth              | Notes |
|--------|-----------------------------------|-------------------|-------|
| GET    | `/api/health`                     | none              | |
| POST   | `/api/signup`                     | none              | `{email, password, role}` |
| POST   | `/api/login`                      | none              | `{email, password}` |
| GET    | `/api/courses`                    | any               | `?mine=1` (instructor only) |
| POST   | `/api/courses`                    | instructor        | `{title, description}` |
| GET    | `/api/courses/<id>`                | any               | |
| PATCH  | `/api/courses/<id>`                | owning instructor | |
| DELETE | `/api/courses/<id>`                | owning instructor | cascades lessons/enrollments/progress |
| POST   | `/api/courses/<id>/enroll`         | student           | idempotent |
| GET    | `/api/courses/<id>/progress`       | enrolled student  | |
| GET    | `/api/courses/<id>/lessons`        | any               | ordered by `order_index` |
| POST   | `/api/courses/<id>/lessons`        | owning instructor | `content_type` must be `text`/`video_url` here |
| GET    | `/api/lessons/<id>`                | any               | |
| PATCH  | `/api/lessons/<id>`                | owning instructor | |
| DELETE | `/api/lessons/<id>`                | owning instructor | |
| POST   | `/api/lessons/<id>/upload`         | owning instructor | multipart `file`; sets `content_type=file` |
| POST   | `/api/lessons/<id>/complete`       | enrolled student  | idempotent |
| GET    | `/api/me/enrollments`              | student           | all enrollments + progress |

## Not in this milestone

Quizzes/assignments, grading, and student/instructor dashboards are
explicitly milestone 2's job per `ARC_QUEUE.md` -- this milestone is
scoped to the data model, CRUD, upload, and progress tracking only.
