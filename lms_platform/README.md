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

133 tests, all passing (91 from milestone 1 + 42 new in milestone 2 --
see below).

## API summary (milestone 1)

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

## Milestone 2 (final): quizzes, assignments, grading, dashboards, certificates

- **Quizzes** (`quizzes`/`quiz_questions`/`quiz_options` tables): an
  instructor authors a quiz in one request --
  `POST /api/courses/<id>/quizzes` with a nested `questions` array,
  each question with a nested `options` array and exactly one
  `is_correct: true` (enforced at creation, `400` otherwise). Students
  see the quiz (`GET /api/quizzes/<id>`) with `is_correct` stripped
  from every option; the owning instructor's view includes it.
  Students submit an attempt (`POST /api/quizzes/<id>/attempts`,
  requires enrollment) with `{"answers": [{"question_id",
  "option_id"}]}`; grading is server-side and pure (`grading.py`,
  no Flask/DB) -- an unanswered question scores as wrong, not a
  validation error, and an option that doesn't belong to the question
  it's paired with is rejected (`400`), same for a duplicate answer to
  one question. A quiz has a `passing_score` (percent); an attempt's
  `passed` flag is `percent >= passing_score`. Instructors list all
  attempts for their quiz (`GET /api/quizzes/<id>/attempts`, with
  student email); students list their own
  (`GET /api/me/quiz-attempts`).
- **Assignments** (`assignments`/`assignment_submissions`): instructor
  creates one (`POST /api/courses/<id>/assignments`, `title` +
  `description` + `max_points` + optional `due_at`). A student submits
  text content (`POST /api/assignments/<id>/submit`, requires
  enrollment); resubmitting **overwrites** the prior submission and
  clears any existing grade/feedback, since new content always needs
  re-grading rather than silently keeping a stale grade attached to it.
  The owning instructor lists submissions
  (`GET /api/assignments/<id>/submissions`, with student email) and
  grades one (`PATCH
  /api/assignments/<id>/submissions/<student_id>/grade`,
  `{"grade", "feedback"}`, bounded to `[0, max_points]`).
- **Certificates** (`certificates` table): a student requests one
  (`POST /api/courses/<id>/certificate`) and is issued a
  `verification_code` (UUID) once eligible per the policy in
  `certificates.py` -- **all** lessons completed, **all** quizzes
  passed (by best attempt), and **all** assignments submitted
  (submission, not a passing grade, is the bar for assignments, since
  grading is instructor-paced and shouldn't gate a certificate on
  turnaround time). A course with none of those trivially qualifies.
  Re-issuing an already-earned certificate is idempotent (`200` with
  the same code, not a new one). Verification
  (`GET /api/certificates/<code>`) is deliberately **public, no
  auth** -- the same way a real credential-verification page works --
  and returns `404` uniformly for a malformed or simply-unknown code.
  `progress_report.py` computes the shared "where does this student
  stand in this course" view used by both the certificate route and
  the dashboard, so the two can't disagree about what counts as done.
- **Dashboards**: one endpoint, `GET /api/me/dashboard`, shaped by
  role. A student gets, per enrolled course: lesson percent complete,
  quizzes passed/total, assignments submitted/total, and whether a
  certificate has been earned (plus its code). An instructor gets, per
  owned course: enrolled student count, quiz count and average
  pass rate (computed off each student's *best* attempt, not every
  retry -- a student who fails then passes counts once, as a pass),
  assignment count, and an ungraded-submissions count -- the thing an
  instructor actually needs to act on next.

## API summary (milestone 2 additions)

| Method | Path                                                          | Auth               | Notes |
|--------|---------------------------------------------------------------|---------------------|-------|
| GET    | `/api/courses/<id>/quizzes`                                   | any                 | question counts only |
| POST   | `/api/courses/<id>/quizzes`                                   | owning instructor   | nested questions + options |
| GET    | `/api/quizzes/<id>`                                           | any                 | `is_correct` hidden from non-owners |
| DELETE | `/api/quizzes/<id>`                                           | owning instructor   | |
| POST   | `/api/quizzes/<id>/attempts`                                  | enrolled student    | auto-graded |
| GET    | `/api/quizzes/<id>/attempts`                                  | owning instructor   | all students' attempts |
| GET    | `/api/me/quiz-attempts`                                       | student             | own attempts, `?quiz_id=` optional |
| GET    | `/api/courses/<id>/assignments`                               | any                 | |
| POST   | `/api/courses/<id>/assignments`                               | owning instructor   | |
| GET    | `/api/assignments/<id>`                                       | any                 | |
| DELETE | `/api/assignments/<id>`                                       | owning instructor   | |
| POST   | `/api/assignments/<id>/submit`                                | enrolled student    | resubmission clears prior grade |
| GET    | `/api/assignments/<id>/submissions`                           | owning instructor   | |
| GET    | `/api/me/submissions`                                         | student             | requires `?assignment_id=` |
| PATCH  | `/api/assignments/<id>/submissions/<student_id>/grade`         | owning instructor   | `{grade, feedback}` |
| POST   | `/api/courses/<id>/certificate`                               | enrolled student    | idempotent once eligible |
| GET    | `/api/certificates/<code>`                                    | **none (public)**   | verification lookup |
| GET    | `/api/me/certificates`                                        | student             | |
| GET    | `/api/me/dashboard`                                           | any                 | shape depends on role |

(Same command runs the full suite -- milestone 1 and milestone 2 tests
live side by side under `tests/`.)
