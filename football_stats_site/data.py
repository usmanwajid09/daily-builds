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

from dataclasses import replace

from .models import Match, Player, Season, Team

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

# Squad composition for the fictional roster generator below: 2 keepers,
# 5 defenders, 5 midfielders, 4 forwards -- a realistic 16-player squad
# shape, not tied to any real club's actual roster.
SQUAD_COMPOSITION: list[tuple[str, int]] = [
    ("GK", 2),
    ("DEF", 5),
    ("MID", 5),
    ("FWD", 4),
]

# Deliberately generic, made-up name parts -- not modeled on any real
# player -- combined to produce plausible-looking fictional full names.
_FIRST_NAMES = [
    "Marcus", "Elias", "Théo", "Kwame", "Diego", "Noah", "Sami", "Luca",
    "Rafael", "Owen", "Kian", "Mateus", "Andrei", "Yusuf", "Hugo",
    "Bram", "Farid", "Callum", "Nico", "Tomas",
]
_LAST_NAMES = [
    "Okafor", "Larsson", "Moreau", "Petrov", "Alves", "Whitfield",
    "Nakamura", "Costa", "Bergman", "Delgado", "Reyes", "Kovac",
    "Bianchi", "Fontaine", "Osei", "Hendricks", "Suárez", "Lindgren",
    "Marchetti", "Voss",
]


def generate_roster(team_name: str, rng: random.Random) -> tuple[Player, ...]:
    """Generate a deterministic fictional squad for one team.

    Names are drawn (with replacement, so occasional repeats across
    different teams are expected and fine) from small generic name-part
    pools -- not real players. Squad numbers are unique within the
    roster and drawn from 1-99. ``team_name`` only seeds nothing extra
    here; determinism comes entirely from the caller-supplied ``rng``,
    so the same ``random.Random`` instance produces a different roster
    per team as it's advanced across calls.
    """
    used_numbers: set[int] = set()

    def next_number() -> int:
        while True:
            n = rng.randint(1, 99)
            if n not in used_numbers:
                used_numbers.add(n)
                return n

    players: list[Player] = []
    for position, count in SQUAD_COMPOSITION:
        for _ in range(count):
            name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
            players.append(Player(name=name, position=position, squad_number=next_number()))
    return tuple(players)



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


# Forwards score most often, then midfielders, then defenders; keepers
# essentially never do (a rare set-piece scramble aside) -- weights are
# illustrative, not modeled on real statistics.
_SCORER_WEIGHT_BY_POSITION = {"FWD": 6, "MID": 3, "DEF": 1, "GK": 0}


def _assign_scorers(
    rng: random.Random, team_name: str, roster: tuple, num_goals: int
) -> list[dict]:
    """Attribute ``num_goals`` goals to players on ``roster``, each with a
    distinct simulated minute (1-90, sorted ascending).

    Weighted toward attacking positions. Falls back to an unweighted
    choice if a roster has no outfield players with positive weight
    (shouldn't happen with ``SQUAD_COMPOSITION``, but keeps this
    function from crashing on a hand-built roster in a test).
    """
    if num_goals == 0 or not roster:
        return []

    weights = [_SCORER_WEIGHT_BY_POSITION.get(p.position, 1) for p in roster]
    if sum(weights) == 0:
        weights = [1] * len(roster)

    minutes = sorted(rng.sample(range(1, 91), k=min(num_goals, 90)))
    # If num_goals > 90 (never happens given the capped goals() model,
    # but guard anyway) pad with 90th-minute goals rather than crashing.
    while len(minutes) < num_goals:
        minutes.append(90)

    events = []
    for minute in minutes:
        scorer = rng.choices(roster, weights=weights, k=1)[0]
        events.append({"team": team_name, "player": scorer.name, "minute": minute})
    return events


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
    if len(set(team_names)) != len(team_names):
        dupes = sorted({n for n in team_names if team_names.count(n) > 1})
        raise ValueError(
            f"generate_season: duplicate team name(s) {dupes} -- each team "
            f"needs a unique name (this used to fail much later, and much "
            f"less clearly, inside Match validation when a duplicate-named "
            f"team was scheduled against itself)"
        )
    schedule = round_robin_schedule(team_names)
    total_matchdays = len(schedule)
    if cutoff_matchday > total_matchdays:
        raise ValueError(
            f"cutoff_matchday ({cutoff_matchday}) exceeds the number of "
            f"matchdays in the schedule ({total_matchdays})"
        )

    rng = random.Random(seed)
    kickoff_slots = ["12:30", "15:00", "17:30", "20:00"]

    # Give every team a deterministic fictional roster (milestone 2:
    # needed for scorer events and the team-detail/search API). Teams
    # passed in already carrying a roster (e.g. a test building its own
    # squads) are left untouched rather than overwritten.
    rostered_teams: list[Team] = [
        t if t.roster else replace(t, roster=generate_roster(t.name, rng))
        for t in teams
    ]
    roster_by_name = {t.name: t.roster for t in rostered_teams}

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
            scorers: list[dict] = []
            if played:
                home_score, away_score = _simulate_score(rng)
                scorers = _assign_scorers(
                    rng, home, roster_by_name.get(home, ()), home_score
                ) + _assign_scorers(
                    rng, away, roster_by_name.get(away, ()), away_score
                )
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
                    scorers=scorers,
                )
            )
            match_id += 1

    return Season(name=name, teams=rostered_teams, matches=matches)
