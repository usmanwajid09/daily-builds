# quiz_api

Quiz API: create multiple-choice quizzes, list them, and submit
answers to get a score with a per-question breakdown. In-memory only
— nothing is written to disk. Correct answers are never exposed by
the read endpoints; they are revealed only in the submit result.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/quizzes` | Create a quiz (`title`, `questions` with `question`, `options`, `correct_index`) |
| GET | `/quizzes` | List all quizzes (no answers) |
| GET | `/quizzes/<id>` | Fetch one quiz (no answers) |
| POST | `/quizzes/<id>/submit` | Submit `answers` (list of option indexes), get score |
| DELETE | `/quizzes/<id>` | Delete a quiz |

Bad payloads (missing title, fewer than 2 options, out-of-range
`correct_index`, wrong answer count) return `400`; unknown quiz ids
return `404`.

## Run

```bash
python -m pytest tests/test_quiz_api.py -q
python quiz_api/demo.py
```

## Demo output

```
Created quiz 1: Python basics (3 questions) -> 201
Quizzes listed: 1
Perfect attempt: 3/3 (100.0%)
Partial attempt: 2/3 (66.7%), per-question: [False, True, True]
Empty questions -> 400: questions must be a non-empty list
Unknown quiz -> 404: quiz 99 not found
```
