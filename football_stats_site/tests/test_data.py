import itertools
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import random

from football_stats_site.data import (
    DEFAULT_TEAMS,
    SQUAD_COMPOSITION,
    generate_roster,
    generate_season,
    round_robin_schedule,
)
from football_stats_site.models import Team


def test_round_robin_requires_even_teams():
    with pytest.raises(ValueError):
        round_robin_schedule(["A", "B", "C"])


def test_round_robin_requires_at_least_two_teams():
    with pytest.raises(ValueError):
        round_robin_schedule(["A"])


def test_round_robin_every_pair_meets_home_and_away_once():
    names = [t.name for t in DEFAULT_TEAMS]
    schedule = round_robin_schedule(names)
    assert len(schedule) == 2 * (len(names) - 1)
    for round_ in schedule:
        assert len(round_) == len(names) // 2

    pairs = Counter()
    for round_ in schedule:
        for home, away in round_:
            pairs[(home, away)] += 1

    for a, b in itertools.permutations(names, 2):
        assert pairs[(a, b)] == 1, f"{a} vs {b} should occur exactly once"


def test_round_robin_no_team_plays_itself_or_twice_in_a_round():
    names = [t.name for t in DEFAULT_TEAMS]
    schedule = round_robin_schedule(names)
    for round_ in schedule:
        appearances = Counter()
        for home, away in round_:
            assert home != away
            appearances[home] += 1
            appearances[away] += 1
        assert all(count == 1 for count in appearances.values()), (
            "every team should appear exactly once per matchday"
        )


def test_generate_season_is_deterministic_for_same_seed():
    s1 = generate_season(seed=42)
    s2 = generate_season(seed=42)
    d1 = [m.to_dict() for m in s1.matches]
    d2 = [m.to_dict() for m in s2.matches]
    assert d1 == d2


def test_generate_season_different_seed_changes_scores():
    s1 = generate_season(seed=1)
    s2 = generate_season(seed=2)
    scores1 = [(m.home_score, m.away_score) for m in s1.matches if m.is_played]
    scores2 = [(m.home_score, m.away_score) for m in s2.matches if m.is_played]
    assert scores1 != scores2


def test_generate_season_cutoff_splits_played_vs_scheduled():
    season = generate_season(cutoff_matchday=5)
    for m in season.matches:
        if m.matchday <= 5:
            assert m.is_played, f"matchday {m.matchday} should be played"
        else:
            assert not m.is_played, f"matchday {m.matchday} should be a fixture"


def test_generate_season_cutoff_beyond_schedule_length_raises():
    with pytest.raises(ValueError):
        generate_season(teams=DEFAULT_TEAMS, cutoff_matchday=999)


def test_generate_season_rejects_odd_team_count():
    odd_teams = DEFAULT_TEAMS[:9]
    with pytest.raises(ValueError):
        generate_season(teams=odd_teams)


def test_generate_season_rejects_duplicate_team_names_with_clear_error():
    # Regression: this used to fail late and confusingly inside Match's
    # own validation ("home_team and away_team must differ") once the
    # schedule happened to pair a duplicate-named team against itself,
    # instead of failing immediately with a clear message.
    dupes = [
        Team("A FC", "AAA"),
        Team("A FC", "BBB"),
        Team("C FC", "CCC"),
        Team("D FC", "DDD"),
    ]
    with pytest.raises(ValueError, match="duplicate team name"):
        generate_season(teams=dupes, cutoff_matchday=2)


def test_generate_season_match_ids_are_unique_and_sequential():
    season = generate_season()
    ids = [m.id for m in season.matches]
    assert ids == list(range(1, len(season.matches) + 1))


def test_generate_season_scores_are_non_negative_and_bounded():
    season = generate_season()
    for m in season.matches:
        if m.is_played:
            assert 0 <= m.home_score <= 8
            assert 0 <= m.away_score <= 8


def test_generate_roster_has_expected_composition_and_unique_numbers():
    roster = generate_roster("Test FC", random.Random(1))
    expected_size = sum(count for _pos, count in SQUAD_COMPOSITION)
    assert len(roster) == expected_size

    positions = Counter(p.position for p in roster)
    for pos, count in SQUAD_COMPOSITION:
        assert positions[pos] == count

    numbers = [p.squad_number for p in roster]
    assert len(numbers) == len(set(numbers)), "squad numbers must be unique"
    assert all(1 <= n <= 99 for n in numbers)


def test_generate_roster_deterministic_for_same_rng_state():
    r1 = generate_roster("Team A", random.Random(42))
    r2 = generate_roster("Team A", random.Random(42))
    assert [(p.name, p.position, p.squad_number) for p in r1] == [
        (p.name, p.position, p.squad_number) for p in r2
    ]


def test_generate_season_gives_every_team_a_roster():
    season = generate_season()
    for team in season.teams:
        assert len(team.roster) == sum(c for _p, c in SQUAD_COMPOSITION)


def test_generate_season_preserves_caller_supplied_roster():
    from football_stats_site.models import Player

    custom_roster = (Player("Custom Player", "FWD", 99),)
    teams = [Team(t.name, t.short_code, roster=custom_roster if i == 0 else ())
             for i, t in enumerate(DEFAULT_TEAMS)]
    season = generate_season(teams=teams)
    assert season.teams[0].roster == custom_roster
    # every other team still got an auto-generated roster, not left empty
    assert all(len(t.roster) > 0 for t in season.teams[1:])


def test_generate_season_scorer_goal_counts_match_recorded_score():
    season = generate_season()
    played = [m for m in season.matches if m.is_played]
    assert played, "expected at least one played match for this test to mean anything"
    for m in played:
        home_goals = sum(1 for e in m.scorers if e["team"] == m.home_team)
        away_goals = sum(1 for e in m.scorers if e["team"] == m.away_team)
        assert home_goals == m.home_score, m
        assert away_goals == m.away_score, m
        # scorers must belong to the scoring team's actual roster
        home_names = {p.name for p in season.find_team(m.home_team).roster}
        away_names = {p.name for p in season.find_team(m.away_team).roster}
        for event in m.scorers:
            if event["team"] == m.home_team:
                assert event["player"] in home_names
            else:
                assert event["player"] in away_names
            assert 1 <= event["minute"] <= 90


def test_generate_season_unplayed_matches_have_no_scorers():
    season = generate_season()
    fixtures = [m for m in season.matches if not m.is_played]
    assert fixtures
    assert all(m.scorers == [] for m in fixtures)


def test_generate_season_is_fully_deterministic_including_rosters_and_scorers():
    s1 = generate_season()
    s2 = generate_season()
    r1 = [[(p.name, p.position, p.squad_number) for p in t.roster] for t in s1.teams]
    r2 = [[(p.name, p.position, p.squad_number) for p in t.roster] for t in s2.teams]
    assert r1 == r2
    assert [m.scorers for m in s1.matches] == [m.scorers for m in s2.matches]
