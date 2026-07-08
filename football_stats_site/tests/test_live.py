import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from football_stats_site.data import generate_season
from football_stats_site.live import MATCH_LENGTH_MINUTES, LiveScoreSimulator


@pytest.fixture
def season():
    return generate_season()


def test_defaults_to_first_unplayed_matchday(season):
    sim = LiveScoreSimulator(season)
    unplayed = sorted({m.matchday for m in season.matches if not m.is_played})
    assert sim.matchday == unplayed[0]


def test_explicit_played_matchday_raises_no_such_matchday_message():
    from football_stats_site.models import Match, Season, Team

    a, b = Team("A FC", "AFC"), Team("B FC", "BFC")
    season = Season(
        name="Tiny",
        teams=[a, b],
        matches=[Match(1, 1, "2026-01-01", "15:00", "A FC", "B FC", home_score=1, away_score=0)],
    )
    with pytest.raises(ValueError):
        LiveScoreSimulator(season)  # no unplayed matchday at all


def test_unknown_matchday_raises(season):
    with pytest.raises(ValueError):
        LiveScoreSimulator(season, matchday=9999)


def test_before_any_tick_status_is_scheduled_with_no_score(season):
    sim = LiveScoreSimulator(season)
    snap = sim.snapshot()
    assert snap["clock_minute"] == 0
    assert snap["finished"] is False
    for m in snap["matches"]:
        assert m["status"] == "SCHEDULED"
        assert m["home_score"] is None
        assert m["away_score"] is None
        assert m["scorers"] == []


def test_minute_advances_monotonically_and_clamps_at_90(season):
    sim = LiveScoreSimulator(season, minutes_per_tick=13)
    minutes_seen = [0]
    for _ in range(20):
        sim.tick()
        minutes_seen.append(sim.minute)
    assert minutes_seen == sorted(minutes_seen)
    assert sim.minute == MATCH_LENGTH_MINUTES
    assert sim.is_finished


def test_tick_after_finished_is_a_noop(season):
    sim = LiveScoreSimulator(season, minutes_per_tick=90)
    sim.tick()
    assert sim.is_finished
    snap_a = sim.snapshot()
    sim.tick()
    snap_b = sim.snapshot()
    assert snap_a == snap_b


def test_scores_only_increase_or_stay_the_same_as_minutes_advance(season):
    sim = LiveScoreSimulator(season, minutes_per_tick=5)
    prev_scores = {m["id"]: (0, 0) for m in sim.snapshot()["matches"]}
    while not sim.is_finished:
        sim.tick()
        snap = sim.snapshot()
        for m in snap["matches"]:
            ph, pa = prev_scores[m["id"]]
            assert m["home_score"] >= ph
            assert m["away_score"] >= pa
            prev_scores[m["id"]] = (m["home_score"], m["away_score"])


def test_final_snapshot_matches_planned_final_score(season):
    sim = LiveScoreSimulator(season, minutes_per_tick=7)
    while not sim.is_finished:
        sim.tick()
    snap = sim.snapshot()
    for plan, m in zip(sim._plans, snap["matches"]):
        assert m["status"] == "FT"
        assert m["home_score"] == plan.final_home_score
        assert m["away_score"] == plan.final_away_score


def test_scorers_are_drawn_from_the_scoring_teams_actual_roster(season):
    sim = LiveScoreSimulator(season, minutes_per_tick=90)
    sim.tick()
    snap = sim.snapshot()
    for m in snap["matches"]:
        home_names = {p.name for p in season.find_team(m["home_team"]).roster}
        away_names = {p.name for p in season.find_team(m["away_team"]).roster}
        for event in m["scorers"]:
            if event["team"] == m["home_team"]:
                assert event["player"] in home_names
            else:
                assert event["player"] in away_names


def test_two_simulators_same_seed_produce_identical_snapshots(season):
    sim1 = LiveScoreSimulator(season, seed=99, minutes_per_tick=10)
    sim2 = LiveScoreSimulator(season, seed=99, minutes_per_tick=10)
    for _ in range(9):
        sim1.tick()
        sim2.tick()
        assert sim1.snapshot() == sim2.snapshot()


def test_reset_rewinds_clock_but_keeps_the_same_plan(season):
    sim = LiveScoreSimulator(season, minutes_per_tick=15)
    for _ in range(3):
        sim.tick()
    mid_snapshot = sim.snapshot()
    sim.reset()
    assert sim.minute == 0
    for _ in range(3):
        sim.tick()
    assert sim.snapshot() == mid_snapshot
