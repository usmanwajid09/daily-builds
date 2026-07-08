import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from football_stats_site.models import Match
from football_stats_site.standings import compute_standings


def test_standings_ignores_unplayed_matches():
    matches = [
        Match(1, 1, "2026-01-01", "15:00", "A", "B", home_score=2, away_score=0),
        Match(2, 2, "2026-01-08", "15:00", "A", "B"),  # unplayed
    ]
    table = compute_standings(matches, ["A", "B"])
    by_team = {row["team"]: row for row in table}
    assert by_team["A"]["played"] == 1
    assert by_team["B"]["played"] == 1


def test_standings_win_draw_loss_points():
    matches = [
        Match(1, 1, "2026-01-01", "15:00", "A", "B", home_score=2, away_score=0),  # A win
        Match(2, 1, "2026-01-01", "15:00", "C", "D", home_score=1, away_score=1),  # draw
    ]
    table = compute_standings(matches, ["A", "B", "C", "D"])
    by_team = {row["team"]: row for row in table}

    assert by_team["A"]["won"] == 1 and by_team["A"]["points"] == 3
    assert by_team["B"]["lost"] == 1 and by_team["B"]["points"] == 0
    assert by_team["C"]["drawn"] == 1 and by_team["C"]["points"] == 1
    assert by_team["D"]["drawn"] == 1 and by_team["D"]["points"] == 1


def test_standings_team_with_no_matches_still_appears():
    matches = [
        Match(1, 1, "2026-01-01", "15:00", "A", "B", home_score=1, away_score=0),
    ]
    table = compute_standings(matches, ["A", "B", "C"])
    by_team = {row["team"]: row for row in table}
    assert by_team["C"] == {
        "position": by_team["C"]["position"],
        "team": "C",
        "played": 0,
        "won": 0,
        "drawn": 0,
        "lost": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "points": 0,
    }


def test_standings_sort_order_points_then_gd_then_gf_then_name():
    # A and B both have 3 points from one win; A has better GD.
    matches = [
        Match(1, 1, "2026-01-01", "15:00", "A", "X", home_score=3, away_score=0),
        Match(2, 1, "2026-01-01", "15:00", "B", "Y", home_score=1, away_score=0),
        # C and D both draw with 1 point each, same GD(0)/GF(1) -> alphabetical
        Match(3, 1, "2026-01-01", "15:00", "D", "C", home_score=1, away_score=1),
    ]
    table = compute_standings(matches, ["A", "B", "C", "D", "X", "Y"])
    order = [row["team"] for row in table]
    assert order.index("A") < order.index("B")  # better GD ranks first
    # C should sort before D at equal points/GD/GF (alphabetical tie-break)
    c_idx, d_idx = order.index("C"), order.index("D")
    assert c_idx < d_idx


def test_standings_unknown_team_in_match_raises():
    matches = [
        Match(1, 1, "2026-01-01", "15:00", "A", "Ghost", home_score=1, away_score=0),
    ]
    with pytest.raises(ValueError):
        compute_standings(matches, ["A"])


def test_standings_points_total_conserved():
    # Every played match distributes either 3+0 or 1+1 points -- total
    # points across the table should never exceed 3 * played_matches.
    matches = [
        Match(1, 1, "2026-01-01", "15:00", "A", "B", home_score=2, away_score=1),
        Match(2, 1, "2026-01-01", "15:00", "C", "D", home_score=0, away_score=0),
    ]
    table = compute_standings(matches, ["A", "B", "C", "D"])
    total_points = sum(row["points"] for row in table)
    assert total_points == 3 + 2  # one decisive result (3) + one draw (1+1)
