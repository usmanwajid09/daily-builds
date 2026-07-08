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
