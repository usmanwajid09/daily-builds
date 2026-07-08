"""Data ingestion: builds a full league Season.

Real football data sources were evaluated and rejected for this milestone:

* Paid/free REST APIs (e.g. football-data.org) require an API key and are
  not reachable from this sandbox without one.
* Static open-data JSON dumps (e.g. the openfootball/football.json GitHub
  project, served via raw.githubusercontent.com) *are* reachable, but
  downloads of any real season file (tens of KB) reliably stalled or
  truncated through this sandbox's network proxy across multiple retries
  (confirmed: a 46KB fetch came back cut off mid-JSON, and a retry with a
  longer timeout hit the tool's own 45s ceiling without finishing).

So, following the precedent set by the ai-trading-bot arc (which hit the
same kind of network unreliability for market data), this module ships a
seeded, fully deterministic synthetic season generator instead. It is a
realistic stand-in for a real ingestion pipeline: a double round-robin
schedule, plausible fixture dates/kickoff times, and simulated scores for
matches that have already been "played" as of a fixed cutoff matchday.

Swapping in a real data source later only requires replacing
``generate_season`` with a loader that returns the same ``Season`` shape
defined in ``models.py`` -- everything downstream (standings, the API)
is agnostic to where the matches came from.
"""
from __future__ import annotations

import datetime
import random

from .models import Match, Season, Team

# Ten fictional clubs -- deliberately not real teams, to avoid depending on
# (or misrepresenting) any real club's identity, branding, or trademarks.
DEFAULT_TEAMS: list[Team] = [
    Team("River Athletic", "RVR"),
    Team("City Rovers", "CTY"),
    Team("Harborside United", "HBS"),
    Team("Ironbridge FC", "IRB"),
    Team("Northgate Wanderers", "NGW"),
    Team("Vale Park Rangers", "VPR"),
    Team("Whitfield Town", "WHT"),
    Team("Castlemoor FC", "CSM"),
    Team("Redbrook Albion", "RDB"),
    Team("Summit Hill FC", "SHF"),
]

DEFAULT_SEASON_NAME = "Fantasy Premier Division 2025/26"
DEFAULT_SEED = 2026
# Matchdays 1..CUTOFF are treated as already played; the rest are fixtures
# (scheduled, no score yet). Kept as an explicit constant rather than
# derived from wall-clock time so the generated season -- and every test
# against it -- is 100% reproducible regardless of when this runs.
DEFAULT_CUTOFF_MATCHDAY = 11


def round_robin_schedule(team_names: list[str]) -> list[list[tuple[str, str]]]:
    """Double round-robin pairings using the standard "circle" method.

    Returns a list of rounds (matchdays); each round is a list of
    (home, away) pairs. For ``n`` teams this produces ``2 * (n - 1)``
    rounds of ``n // 2`` matches each, with home/away reversed in the
    second half of the season. Requires an even number of teams.
    """
    if len(team_names) < 2:
        raise ValueError("Need at least 2 teams to schedule a season")
    if len(team_names) % 2 != 0:
        raise ValueError(
            f"round_robin_schedule requires an even number of teams, "
            f"got {len(team_names)}"
        )

    teams = list(team_names)
    n = len(teams)
    fixed = teams[0]
    rotating = teams[1:]

    first_half: list[list[tuple[str, str]]] = []
    for _round in range(n - 1):
        pairing = [(fixed, rotating[-1])] if _round % 2 == 0 else [
            (rotating[-1], fixed)
        ]
        others = rotating[:-1]
        half = len(others) // 2
        for i in range(half):
            a, b = others[i], others[-(i + 1)]
            pairing.append((a, b) if (_round + i) % 2 == 0 else (b, a))
        first_half.append(pairing)
        rotating = [rotating[-1]] + rotating[:-1]

    second_half = [[(away, home) for (home, away) in rnd] for rnd in first_half]
    return first_half + second_half


def _simulate_score(rng: random.Random) -> tuple[int, int]:
    """A crude but deterministic goals model: two independent small
    Poisson-ish counts (via repeated coin flips), biased slightly toward
    the home side, capped at a realistic maximum.
    """

    def goals(bias: float) -> int:
        count = 0
        while rng.random() < bias and count < 8:
            count += 1
            bias *= 0.55  # sharply diminishing chance of each extra goal
        return count

    return goals(0.62), goals(0.52)


def generate_season(
    teams: list[Team] | None = None,
    *,
    name: str = DEFAULT_SEASON_NAME,
    seed: int = DEFAULT_SEED,
    cutoff_matchday: int = DEFAULT_CUTOFF_MATCHDAY,
    season_start: datetime.date = datetime.date(2025, 8, 9),
    days_between_matchdays: int = 7,
) -> Season:
    """Generate a full, deterministic synthetic season.

    Matchdays ``1..cutoff_matchday`` get simulated final scores; every
    later matchday is left as an unplayed fixture (``home_score`` /
    ``away_score`` both ``None``).
    """
    teams = list(teams) if teams is not None else list(DEFAULT_TEAMS)
    team_names = [t.name for t in teams]
    schedule = round_robin_schedule(team_names)
    total_matchdays = len(schedule)
    if cutoff_matchday > total_matchdays:
        raise ValueError(
            f"cutoff_matchday ({cutoff_matchday}) exceeds the number of "
            f"matchdays in the schedule ({total_matchdays})"
        )

    rng = random.Random(seed)
    kickoff_slots = ["12:30", "15:00", "17:30", "20:00"]

    matches: list[Match] = []
    match_id = 1
    for matchday_index, pairings in enumerate(schedule, start=1):
        match_date = season_start + datetime.timedelta(
            days=days_between_matchdays * (matchday_index - 1)
        )
        played = matchday_index <= cutoff_matchday
        for home, away in pairings:
            home_score: int | None
            away_score: int | None
            if played:
                home_score, away_score = _simulate_score(rng)
            else:
                home_score, away_score = None, None
            matches.append(
                Match(
                    id=match_id,
                    matchday=matchday_index,
                    date=match_date.isoformat(),
                    kickoff=rng.choice(kickoff_slots),
                    home_team=home,
                    away_team=away,
                    home_score=home_score,
                    away_score=away_score,
                )
            )
            match_id += 1

    return Season(name=name, teams=teams, matches=matches)
