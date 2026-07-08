"""Core data model: teams and matches.

Kept as plain, dependency-free dataclasses so they can be used from the
data generator, the standings calculator, and the API layer without
pulling in an ORM or web framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Team:
    """A club competing in the league."""

    name: str
    short_code: str  # 2-4 letter code, e.g. "RVR" for "River Athletic"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Team.name must not be empty")
        if not (2 <= len(self.short_code) <= 4):
            raise ValueError(
                f"Team.short_code must be 2-4 characters, got {self.short_code!r}"
            )


@dataclass
class Match:
    """A single fixture/result.

    ``home_score``/``away_score`` are ``None`` for matches that have not
    been played yet (a "fixture"); once both are set the match counts as
    a "result" and feeds into the standings table.
    """

    id: int
    matchday: int
    date: str  # ISO date, "YYYY-MM-DD"
    kickoff: str  # "HH:MM", 24h local kickoff time
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    def __post_init__(self) -> None:
        if self.home_team == self.away_team:
            raise ValueError(
                f"Match {self.id}: home_team and away_team must differ "
                f"(got {self.home_team!r} twice)"
            )
        if (self.home_score is None) != (self.away_score is None):
            raise ValueError(
                f"Match {self.id}: home_score and away_score must both be "
                f"set or both be None, got {self.home_score!r}/{self.away_score!r}"
            )
        if self.home_score is not None and self.home_score < 0:
            raise ValueError(f"Match {self.id}: home_score cannot be negative")
        if self.away_score is not None and self.away_score < 0:
            raise ValueError(f"Match {self.id}: away_score cannot be negative")

    @property
    def is_played(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def status(self) -> str:
        return "FT" if self.is_played else "SCHEDULED"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "matchday": self.matchday,
            "date": self.date,
            "kickoff": self.kickoff,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "status": self.status,
        }


@dataclass
class Season:
    """A full league season: the set of teams and all their matches."""

    name: str
    teams: list[Team] = field(default_factory=list)
    matches: list[Match] = field(default_factory=list)

    def team_names(self) -> list[str]:
        return [t.name for t in self.teams]
