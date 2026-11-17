# leaderboard_api

Leaderboard API: submit scores per player and game, get ranked
listings, top-N results, and each player's best score per game.
In-memory only — nothing is written to disk. Ties are broken by
earliest submission: the first player to reach a score ranks above a
later player with the same score.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/scores` | Submit a score (`player`, `game`, `score`) |
| GET | `/leaderboard` | Ranked listing, all games (`?game=<name>`, `?top=N`) |
| GET | `/leaderboard/<game>` | Ranked listing for one game (`?top=N`) |
| GET | `/players/<player>/best` | Best score per game for one player (`?game=<name>`) |
| GET | `/games` | List games that have scores |

Bad payloads (missing player/game/score, non-numeric score, bad
`top`) return `400`; a game or player with no scores returns `404`.

## Run

```bash
python -m pytest tests/test_leaderboard_api.py -q
python leaderboard_api/demo.py
```

## Demo output

```
Submitted: Ayesha scored 1500 in snake (#1) -> 201
Submitted: Bilal scored 2200 in snake (#2) -> 201
Submitted: Ayesha scored 2200 in snake (#3) -> 201
Submitted: Chen scored 900 in snake (#4) -> 201
Submitted: Bilal scored 3100 in tetris (#5) -> 201
Submitted: Ayesha scored 2750 in tetris (#6) -> 201
Snake leaderboard (ties broken by earliest submission):
  #1 Bilal: 2200
  #2 Ayesha: 2200
  #3 Ayesha: 1500
  #4 Chen: 900
Top 2 overall: [('Bilal', 'tetris', 3100), ('Ayesha', 'tetris', 2750)]
Ayesha's best per game: [{'game': 'snake', 'score': 2200}, {'game': 'tetris', 'score': 2750}]
Games tracked: ['snake', 'tetris']
Unknown game -> 404: no scores for game: chess
Missing score -> 400: score is required and must be a number
```
