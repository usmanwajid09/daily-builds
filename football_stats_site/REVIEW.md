# Self-review notes

## Milestone 1 (2026-07-08) -- data model, ingestion, backend API

Read the full branch diff against `main` (13 files, ~1090 lines) before
merging. Found and fixed three real issues, documented below; also
recorded two things that were considered and deliberately left as-is.

### Fixed

1. **404-vs-405 precedence bug in `app.py`'s WSGI dispatcher.** The
   original code checked `method != "GET"` *before* checking whether the
   route existed at all, so `POST /api/totally-not-a-route` returned
   `405 Method Not Allowed` -- implying the route exists but only GET is
   allowed on it, which is wrong; the route never existed. Fixed by
   checking route existence first: unknown routes now always 404
   regardless of method, and 405 is only returned for a real route hit
   with the wrong verb. Added a regression test
   (`test_write_method_on_unknown_route_is_404_not_405`).

2. **`KeyError` crash risk on an unmapped status code.** The status-line
   lookup was a bare `{200: ..., 400: ..., 404: ..., 405: ...}[status]`
   dict subscript -- any handler that ever returned a status not in that
   exact set would crash the request with a raw `KeyError` traceback
   instead of a response. Not currently reachable (every handler only
   returns one of those four codes today), but it's exactly the kind of
   latent landmine that bites the next person who adds a handler.
   Replaced with `.get(status, f"{status} Error")` so an unmapped status
   degrades to a generic-but-valid status line instead of crashing.

3. **Duplicate team names failed late and confusingly.**
   `generate_season(teams=[...])` didn't validate that team names were
   unique. If two teams shared a name, the round-robin schedule would
   eventually pair that name against itself, and the *only* thing that
   caught it was `Match.__post_init__`'s generic
   "home_team and away_team must differ" check -- correct behavior, but
   a confusing error to debug from, since it blames a match ID deep in
   the schedule rather than the actual input problem. Added an explicit
   upfront duplicate-name check in `generate_season` with a clear error
   message, plus a regression test.

### Also fixed (smaller)

- A stale docstring comment on `Team.short_code` said "3-letter code"
  while the actual validation allows 2-4 characters. Corrected the
  comment to match the code.

### Considered and left as-is

- **Team names with spaces in query params** (e.g.
  `?team=River Athletic`) work correctly through `urllib.parse.parse_qs`
  because a real HTTP client/browser URL-encodes the space before it
  reaches the server. It's only ambiguous if someone hand-types an
  unencoded URL in a raw socket client. Documented the expected
  URL-encoding behavior implicitly via the `curl "..."` (quoted)
  examples in the README rather than adding server-side leniency for an
  invalid input format.
- **No persistence / in-memory only.** Deliberate for this milestone --
  the API always serves the same seeded season on every process
  restart. Called out explicitly as a limitation in the README rather
  than silently treated as a non-issue.
- **Tie-break rule (points/GD/GF/alphabetical) has no head-to-head or
  playoff logic.** Real leagues do; out of scope for milestone 1, and
  explicitly documented in both `standings.py`'s docstring and the
  README rather than left implicit.

### What's genuinely synthetic and why (carried into REVIEW for
### visibility, full detail in README)

Two real external data sources were tried before falling back to a
generator: a real REST API (needs an API key not available here) and a
real static JSON dataset on `raw.githubusercontent.com` (reachable, but
every attempt to download a real season file -- tens of KB -- stalled
or arrived truncated through this sandbox's network proxy, even with an
extended timeout). This mirrors the exact failure mode the
`ai-trading-bot` arc hit with real market-data APIs, so the same
"seeded synthetic generator, clearly labeled" strategy was reused here.
The generator's correctness (every team plays every other team exactly
once home and once away) is verified combinatorially in
`tests/test_data.py`, not just spot-checked.

Total: 39 tests (`pytest football_stats_site/tests/ -v`), all passing.

## Milestone 2 (2026-07-08) -- rosters/scorers, live simulation, new API routes, frontend

Read the full branch diff against `main` (19 files, ~2150 lines added)
before merging, and ran `pyflakes` over every module. Found and fixed
three real issues.

### Fixed

1. **Invalid JSON on a 404 whenever the request path contained an
   apostrophe (`static_site.py`).** The static file server's error
   helper built its JSON body by hand with
   `f'{{"error": {message!r}}}'.replace("'", '"')` instead of
   `json.dumps`. Python's `repr()` switches to double-quote delimiters
   (without escaping) when a string itself contains a single quote, so
   for a request like `GET /don't-exist.js` the blanket
   `.replace("'", '"')` turned that *inner* apostrophe into a stray,
   unescaped double quote -- producing a response body that isn't valid
   JSON at all (confirmed: `json.loads` raises
   `Expecting ',' delimiter`). Any JSON-parsing client hitting a 404 for
   a path with an apostrophe in it would get a parse error instead of a
   clean error message. Fixed by using `json.dumps({"error": message})`
   like `app.py` already does, and added a regression test
   (`test_404_error_body_is_valid_json_even_when_path_contains_an_apostrophe`).

2. **Unused import (`pyflakes`-caught).** `live.py` imported
   `dataclasses.field` but never used it (the `_LiveMatchPlan` dataclass
   only has required fields). Removed the dead import.

3. **Stale API docstring (same class of bug flagged in milestone 1's
   review).** `app.py`'s module docstring still only listed milestone
   1's six routes; the five new milestone-2 routes
   (`/api/teams/<name>`, `/api/players`, `/api/search`, `/api/top-scorers`,
   `/api/live`) were undocumented at the top of the file, even though
   they were fully documented in the README. Updated the docstring to
   list every route and the current error-status behavior.

### Also fixed (smaller, hardening)

- `standings.top_scorers` sorted by `(-goals, player_name)`, which would
  raise `TypeError: '<' not supported between instances of 'NoneType'
  and 'str'` if any scorer event ever had `player: None` (e.g. an empty
  roster edge case). Not reachable via `generate_season()` today (every
  team always gets a 16-player roster), but `top_scorers` is a general
  utility over `Match.scorers`, and `live.py`'s `_pick_scorer` *does*
  return `None` for an empty roster in principle. Changed the sort key
  to `player_name or ""` and added a regression test so this stays
  fixed if a future caller ever hits the edge case.

### Considered and left as-is

- **`LiveScoreSimulator.minutes_per_tick` has no validation** (e.g. `0`
  would mean the clock never advances). Not reachable through the
  public API today -- `/api/live` always constructs the simulator with
  the default -- so left unvalidated rather than adding a guard for an
  input path that doesn't exist yet. Would need revisiting if
  `minutes_per_tick` is ever exposed as a query parameter.
- **The scorer-position weighting table
  (`{"FWD": 6, "MID": 3, "DEF": 1, "GK": 0}`) is duplicated** between
  `data.py` (historical goals) and `live.py` (simulated live goals).
  Considered extracting to a shared constant, but the two modules
  intentionally have separate, decoupled simulation models (season
  generation happens once at startup; live simulation is a completely
  separate deterministic plan keyed off a different seed) and forcing a
  shared import would couple them for a two-line table. Left as
  intentional, minor duplication.
- **Live-simulated score changes aren't reflected in
  `/api/matches`/`/api/fixtures`/`/api/results`.** By design: the live
  simulator never mutates `Season.matches`, so while a match is "live"
  it still shows as `SCHEDULED` with no score everywhere except the
  dedicated `/api/live` endpoint and the frontend's Live tab. This is
  documented as a limitation in the README rather than silently
  papered over with a fake merge.
- **No client-side routing / deep links** in the vanilla-JS frontend --
  a bookmarked or refreshed URL for a specific team doesn't restore
  that view; the static server's SPA fallback just serves the Table tab
  shell. Documented as a limitation in the README; adding real routing
  would mean either a hash-router or history API integration, which
  felt like scope creep for a build with no framework and no build
  step.

Total: 89 tests (`pytest football_stats_site/tests/ -v`), all passing,
plus a clean `pyflakes` pass. Also manually verified end-to-end against
a running `wsgiref` dev server via `curl`: `index.html`/`styles.css`/
`app.js` are served byte-for-byte identical to the source files, every
new API route responds with the expected shape, and requesting an
unknown static asset correctly 404s.
