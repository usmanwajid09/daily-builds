# football-stats-site

A fotmob-style football stats backend: league standings, fixtures, and
results, served over a small JSON API. **Milestone 1** of a two-day arc
(this milestone is the data model + ingestion + backend API; milestone 2
adds a frontend UI and simulated live-score polling).

## Data source: synthetic, seeded, and documented

This milestone does **not** use a real football data API or dataset.
Two real options were tried and both failed from this sandbox:

1. **REST APIs** (e.g. football-data.org) require an API key not
   available here.
2. **Static open-data JSON** (openfootball/football.json on
   `raw.githubusercontent.com`) *is* reachable, but downloading any real
   season file (tens of KB) reliably stalled or arrived truncated through
   this sandbox's network proxy, across multiple retries and timeouts.

So, following the same precedent as the `ai-trading-bot` arc (which hit
identical unreliability fetching real market data), `data.py` ships a
**seeded, fully deterministic synthetic season generator**: 10 fictional
clubs (not real teams -- no trademark/branding concerns), a real
double round-robin schedule (every team plays every other team home and
away, verified by tests), and simulated match scores for everything up
to a fixed cutoff matchday, with the remaining matchdays left as
unplayed fixtures. Swapping in a real data source later is a drop-in
replacement for `generate_season()` -- everything downstream (standings,
the API) only depends on the `Season`/`Match`/`Team` shapes in
`models.py`, not on where the data came from.

## Data model (`models.py`)

- `Team(name, short_code)` -- a club.
- `Match(id, matchday, date, kickoff, home_team, away_team, home_score,
  away_score)` -- a fixture; `home_score`/`away_score` are both `None`
  until played (`status == "SCHEDULED"`), then both set
  (`status == "FT"`). Validated: no team plays itself, scores are
  non-negative, and score fields can't be half-set.
- `Season(name, teams, matches)` -- the full league.

## Standings (`standings.py`)

`compute_standings(matches, team_names)` builds a league table from
played matches only (3 points for a win, 1 for a draw), sorted by
points, then goal difference, then goals for, then team name
alphabetically as a final deterministic tie-break. Real leagues use
head-to-head records or playoffs for ties -- out of scope here, and
called out explicitly rather than silently approximated.

## API (`app.py`, stdlib-only WSGI, no Flask/FastAPI)

Read-only JSON API implemented directly on `wsgiref` -- no third-party
web framework. This matches the rest of the repo's bias toward minimal
dependencies (numpy-only GPT, from-scratch trading indicators) and
sidesteps this sandbox's unreliable `pip install` for new packages.

| Route | Description |
|---|---|
| `GET /api/health` | `{"status": "ok"}` |
| `GET /api/teams` | All teams: `{"teams": [{"name", "short_code"}, ...]}` |
| `GET /api/standings` | League table, see shape above |
| `GET /api/matches` | All matches. Filters: `?matchday=N`, `?team=Name` |
| `GET /api/fixtures` | Unplayed matches only (same filters) |
| `GET /api/results` | Played matches only (same filters) |

Errors: unknown route -> 404, non-GET method -> 405 (API is read-only),
non-integer `matchday` -> 400, unknown `team` -> 404 with the list of
known team names.

## Running it

```bash
# from the repo root
python3 -m football_stats_site.server        # serves on :8000
python3 -m football_stats_site.server 9000    # or a custom port

curl http://127.0.0.1:8000/api/standings
curl "http://127.0.0.1:8000/api/matches?team=River Athletic"
curl "http://127.0.0.1:8000/api/fixtures?matchday=15"
```

## Tests

```bash
python3 -m pytest football_stats_site/tests/ -v
```

37 tests: data-model validation, round-robin schedule correctness
(every team plays every other team exactly once home and once away,
verified combinatorially), standings computation (points, tie-breaks,
teams with zero matches still listed), and the full API surface
(happy paths, filters, and every error status).

## Limitations (milestone 1)

- Synthetic data, not a real league -- see "Data source" above.
- No frontend yet (milestone 2).
- No live-score updates -- the season is generated once at process
  start and never changes while the server runs (milestone 2 adds
  simulated live-score polling).
- Tie-break rule is points/GD/GF/name only; no head-to-head or playoff
  logic.
- No persistence layer -- everything is in memory, regenerated from the
  same seed on every server restart.
