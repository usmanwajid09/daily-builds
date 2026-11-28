# trivia_api

A small Flask trivia API: seeded questions across science, history,
geography, sports and technology (easy, medium and hard), answering
with per-player scores, and a leaderboard — kept in memory (a fresh
store per `create_app()` call, nothing written to disk).

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| GET | `/questions` | List questions (no answers); `?category=` and `?difficulty=` filters |
| GET | `/questions/<id>` | Fetch one question (no answer) |
| POST | `/questions` | Add a question: `{"question", "options" (>= 2), "answer_index", "category", "difficulty"}` |
| POST | `/answer` | Answer: `{"player", "question_id", "answer_index"}` — returns `{"correct", "correct_index", "player", "score", "answered"}` |
| GET | `/scores` | Leaderboard, score desc then player name asc, with accuracy |
| GET | `/scores/<player>` | One player's score record |

Validation: an invalid difficulty filter or invalid question/answer
fields return JSON `400`; an unknown question or a player who never
answered returns JSON `404` (missing `player` on `/answer` is also
`404`). A player may answer the same question multiple times — each
attempt counts toward their score and answered total.

## Run

```bash
python trivia_api/demo.py
python -m pytest tests/test_trivia_api.py -v
```

## Example output

```
Seeded questions: 16
First question: What is the chemical symbol for water? [science/easy]
Science questions: 4
Added question #17: What is the largest ocean on Earth? (answer: Pacific)
Usman answered correctly: True (score 1/1)
Usman answered correctly: False (score 1/2)
Leaderboard: [('Usman', 1, 2, 0.5)]
```
