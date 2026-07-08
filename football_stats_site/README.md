# football-stats-site

A fotmob-style football stats site: league standings, fixtures/results,
team pages, a top-scorers table, and simulated live scores, with a JSON
API backend and a vanilla HTML/CSS/JS frontend. Two-day arc, both
milestones complete: **milestone 1** (data model + ingestion + backend
API) and **milestone 2** (rosters/scorer events, the live-score
simulator, more API endpoints, and the frontend UI).

## Quickstart

```bash
# from the repo root
python3 -m football_stats_site.server        # serves on :8000
# open http://127.0.0.1:8000/ in a browser for the UI,
# or curl http://127.0.0.1:8000/api/standings for raw JSON
```

## Data source: synthetic, seeded, and documented

Neither milestone uses a real football data API or dataset.
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

- `Player(name, position, squad_number)` -- a fictional squad member
  (milestone 2). `position` is one of `GK`/`DEF`/`MID`/`FWD`.
- `Team(name, short_code, roster=())` -- a club; `roster` is a tuple of
  `Player`. It's excluded from `Team`'s equality/hash (`compare=False,
  hash=False`) so `Team` stays usable as a dict/set key exactly as in
  milestone 1, regardless of roster contents.
- `Match(id, matchday, date, kickoff, home_team, away_team, home_score,
  away_score, scorers=[])` -- a fixture; `home_score`/`away_score` are
  both `None` until played (`status == "SCHEDULED"`), then both set
  (`status == "FT"`). Validated: no team plays itself, scores are
  non-negative, and score fields can't be half-set. `scorers` (milestone
  2) is a list of `{"team", "player", "minute"}` goal events, empty for
  unplayed matches.
- `Season(name, teams, matches)` -- the full league. `find_team(name)`
  (milestone 2) looks up a `Team` by name.

## Rosters and goal events (`data.py`)

Milestone 2 extends the synthetic generator: every team gets a
deterministic 16-player fictional squad (2 GK / 5 DEF / 5 MID / 4 FWD,
names drawn from small generic name-part pools -- not real players),
and every played match's goals are attributed to individual scorers
(weighted toward attacking positions) with a simulated minute. Verified
by tests: scorer goal counts always sum back to the match's recorded
score, every scorer is actually on the scoring team's roster, and the
whole thing (rosters + scorers, on top of milestone 1's schedule +
scores) stays fully deterministic for a given seed.

## Live-score simulation (`live.py`)

There's no real live-data feed available here (same sandbox network
constraints as the historical data -- see "Data source" above), so
`LiveScoreSimulator` simulates one matchday "going live" instead: a
final score and full set of goal events are pre-planned deterministically
for that matchday's matches, but only revealed incrementally as an
explicit `tick()` call advances a simulated match clock. The
`/api/live` route calls `tick()` once per request, so **polling the
endpoint is what advances the game** -- a client repeatedly hitting
`/api/live` sees the score progress exactly like it would against a
real live-score API, just without any real-time/wall-clock dependency
(fully deterministic, no `sleep()`, fast and non-flaky to test). If the
season has no unplayed matchday to simulate, the app degrades
gracefully: `/api/live` still returns 200 with an empty match list and
an explanatory `note`, rather than failing to start.

## Standings (`standings.py`)

`compute_standings(matches, team_names)` builds a league table from
played matches only (3 points for a win, 1 for a draw), sorted by
points, then goal difference, then goals for, then team name
alphabetically as a final deterministic tie-break. Real leagues use
head-to-head records or playoffs for ties -- out of scope here, and
called out explicitly rather than silently approximated.

`top_scorers(matches, limit=None)` (milestone 2) folds every match's
`scorers` events into a golden-boot ranking (goals desc, player name
asc tie-break).

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
| `GET /api/teams/<name>` | Team detail: roster, standings row, last-5 form, next-5 fixtures (milestone 2) |
| `GET /api/players` | All players. Filters: `?team=Name`, `?q=`/`?search=` substring on name (milestone 2) |
| `GET /api/search?q=` | Combined team + player substring search for the frontend search box (milestone 2) |
| `GET /api/top-scorers` | Golden boot ranking. Optional `?limit=N` (milestone 2) |
| `GET /api/live` | Simulated live matchday; **each GET call advances the clock** -- see "Live-score simulation" above (milestone 2) |

Errors: unknown route -> 404, non-GET method -> 405 (API is read-only),
non-integer `matchday`/`limit` -> 400, unknown `team` -> 404 with the
list of known team names (including for `/api/teams/<name>` and
`/api/players?team=`).

## Frontend (`static/`, served by `web.py`'s `CombinedApp`)

A single-page vanilla HTML/CSS/JS frontend -- no build step, no
framework, just `fetch()` and plain DOM APIs, matching the backend's
stdlib-only bias. `web.py` composes the JSON API with a small
dependency-free static file server (`static_site.py`, with
path-traversal protection and an SPA-style fallback to `index.html` for
extensionless paths) so one `wsgiref` process serves both.

Tabs: **Table** (standings, click a row for team detail), **Fixtures &
Results** (match cards grouped by matchday, with a matchday filter and
per-match scorer/minute lines for played matches), **Live** (polls
`/api/live` every 3 seconds -- each poll both advances the
server-side simulated clock *and* updates the DOM, so this is a real
end-to-end simulated live experience, not just a client-side
animation), and **Top Scorers**. A debounced search box in the top bar
hits `/api/search` across both teams and players and jumps straight to
the matching team's page. Team pages show a standings snapshot, last-5
form pills, upcoming fixtures, and the full squad grouped by position.

## Running it

```bash
python3 -m football_stats_site.server         # serves on :8000
python3 -m football_stats_site.server 9000     # or a custom port

curl http://127.0.0.1:8000/api/standings
curl "http://127.0.0.1:8000/api/matches?team=River Athletic"
curl "http://127.0.0.1:8000/api/fixtures?matchday=15"
curl "http://127.0.0.1:8000/api/teams/River Athletic"
curl "http://127.0.0.1:8000/api/search?q=river"
curl http://127.0.0.1:8000/api/live   # poll repeatedly to watch it advance
```

## Tests

```bash
python3 -m pytest football_stats_site/tests/ -v
```

87 tests: everything from milestone 1 (data-model validation,
round-robin schedule correctness verified combinatorially, standings
computation) plus milestone 2's roster/scorer generation (goal counts
always match recorded scores, scorers always drawn from the actual
roster, full determinism), the live-score simulator (monotonic clock
and scores, deterministic final score, graceful no-live-matchday
handling), `top_scorers` ranking, every new API route (including the
dynamic `/api/teams/<name>` route's 404-vs-405 precedence), and the
static file server (path traversal blocked, SPA fallback, content
types, 404 vs. 405).

## Limitations

- Synthetic data throughout, not a real league or real players -- see
  "Data source" and "Rosters and goal events" above.
- Live scores are simulated (deterministic, poll-driven) rather than a
  real real-time feed -- see "Live-score simulation" above. Only one
  matchday can be "live" at a time (the first unplayed one), and the
  live clock/scores reset if the server process restarts.
- Tie-break rule is points/GD/GF/name only; no head-to-head or playoff
  logic.
- No persistence layer -- everything is in memory, regenerated from the
  same seed on every server restart (including rosters and the live
  simulator's plan).
- The frontend is intentionally framework-free and single-page; there's
  no client-side routing (deep links to a specific team don't restore
  that view on reload -- they fall back to the Table tab via the static
  server's SPA fallback), and no build/bundle/minify step.
- No auth, no write endpoints, no rate limiting -- this is a read-only
  demo API.
