import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from football_stats_site.models import Match, Player, Season, Team


def test_team_requires_name():
    with pytest.raises(ValueError):
        Team("", "ABC")


def test_team_short_code_length_validated():
    with pytest.raises(ValueError):
        Team("Some FC", "A")
    with pytest.raises(ValueError):
        Team("Some FC", "ABCDE")
    Team("Some FC", "AB")  # 2 chars ok
    Team("Some FC", "ABCD")  # 4 chars ok


def test_match_rejects_same_team_twice():
    with pytest.raises(ValueError):
        Match(1, 1, "2026-01-01", "15:00", "A FC", "A FC")


def test_match_rejects_partial_score():
    with pytest.raises(ValueError):
        Match(1, 1, "2026-01-01", "15:00", "A FC", "B FC", home_score=1, away_score=None)
    with pytest.raises(ValueError):
        Match(1, 1, "2026-01-01", "15:00", "A FC", "B FC", home_score=None, away_score=1)


def test_match_rejects_negative_score():
    with pytest.raises(ValueError):
        Match(1, 1, "2026-01-01", "15:00", "A FC", "B FC", home_score=-1, away_score=0)


def test_match_is_played_and_status():
    scheduled = Match(1, 1, "2026-01-01", "15:00", "A FC", "B FC")
    assert not scheduled.is_played
    assert scheduled.status == "SCHEDULED"

    played = Match(2, 1, "2026-01-01", "15:00", "A FC", "B FC", home_score=2, away_score=0)
    assert played.is_played
    assert played.status == "FT"


def test_match_to_dict_roundtrip_shape():
    m = Match(7, 3, "2026-02-01", "20:00", "A FC", "B FC", home_score=1, away_score=1)
    d = m.to_dict()
    assert d == {
        "id": 7,
        "matchday": 3,
        "date": "2026-02-01",
        "kickoff": "20:00",
        "home_team": "A FC",
        "away_team": "B FC",
        "home_score": 1,
        "away_score": 1,
        "status": "FT",
        "scorers": [],
    }


def test_player_validates_position_and_squad_number():
    p = Player("Alex Rivera", "MID", 8)
    assert p.position == "MID"
    with pytest.raises(ValueError):
        Player("Bad Pos", "STRIKER", 9)
    with pytest.raises(ValueError):
        Player("Bad Number", "GK", 0)
    with pytest.raises(ValueError):
        Player("Bad Number", "GK", 100)
    with pytest.raises(ValueError):
        Player("  ", "GK", 1)


def test_team_roster_defaults_empty_and_does_not_affect_equality_or_hash():
    bare = Team("Roster FC", "RFC")
    assert bare.roster == ()

    stocked = Team("Roster FC", "RFC", roster=(Player("Sam Lee", "FWD", 9),))
    # roster is compare=False/hash=False -- two Teams with the same
    # name/short_code are still equal and hash the same regardless of
    # roster contents, so Team stays usable as a dict/set key exactly as
    # in milestone 1.
    assert bare == stocked
    assert hash(bare) == hash(stocked)
    assert stocked.roster[0].name == "Sam Lee"


def test_match_to_dict_includes_scorers():
    m = Match(
        1, 1, "2026-01-01", "15:00", "A FC", "B FC",
        home_score=1, away_score=0,
        scorers=[{"team": "A FC", "player": "Sam Lee", "minute": 23}],
    )
    d = m.to_dict()
    assert d["scorers"] == [{"team": "A FC", "player": "Sam Lee", "minute": 23}]
    # to_dict returns a copy, not the live list
    d["scorers"].append({"team": "X", "player": "Y", "minute": 1})
    assert len(m.scorers) == 1


def test_season_find_team():
    a = Team("A FC", "AFC")
    b = Team("B FC", "BFC")
    season = Season(name="Test League", teams=[a, b], matches=[])
    assert season.find_team("A FC") is a
    assert season.find_team("Nonexistent FC") is None
