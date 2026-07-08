import itertools
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from football_stats_site.data import (
    DEFAULT_TEAMS,
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
