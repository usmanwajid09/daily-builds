"""Standings (league table) computation from a list of matches."""
from __future__ import annotations

from dataclasses import dataclass

from .models import Match


@dataclass
class TeamRecord:
    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def points(self) -> int:
        return self.won * 3 + self.drawn

    def to_dict(self, position: int) -> dict:
        return {
            "position": position,
            "team": self.team,
            "played": self.played,
            "won": self.won,
            "drawn": self.drawn,
            "lost": self.lost,
            "goals_for": self.goals_for,
            "goals_against": self.goals_against,
            "goal_difference": self.goal_difference,
            "points": self.points,
        }


def compute_standings(matches: list[Match], team_names: list[str]) -> list[dict]:
    """Compute a sorted league table from played matches only.

    Ranking order: points desc, goal difference desc, goals for desc,
    team name asc (alphabetical is an arbitrary but fully deterministic
    final tie-break -- real leagues use head-to-head/playoffs, which is
    out of scope for this milestone and noted in the README).

    ``team_names`` is required (rather than inferred from matches) so
    that a team with zero played matches so far still appears in the
    table with all-zero stats, instead of silently vanishing.
    """
    records = {name: TeamRecord(team=name) for name in team_names}

    for match in matches:
        if not match.is_played:
            continue
        for side in (match.home_team, match.away_team):
            if side not in records:
                raise ValueError(
                    f"Match {match.id} references team {side!r} which is "
                    f"not in team_names"
                )

        home = records[match.home_team]
        away = records[match.away_team]

        home.played += 1
        away.played += 1
        home.goals_for += match.home_score
        home.goals_against += match.away_score
        away.goals_for += match.away_score
        away.goals_against += match.home_score

        if match.home_score > match.away_score:
            home.won += 1
            away.lost += 1
        elif match.home_score < match.away_score:
            away.won += 1
            home.lost += 1
        else:
            home.drawn += 1
            away.drawn += 1

    ordered = sorted(
        records.values(),
        key=lambda r: (-r.points, -r.goal_difference, -r.goals_for, r.team),
    )
    return [r.to_dict(position=i + 1) for i, r in enumerate(ordered)]
