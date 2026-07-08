"""Simulated live-score engine.

Real live-score data would come from a sports-data websocket/polling
feed. There's no such feed available here (see data.py for the same
constraint on historical data), so this module simulates one matchday
"going live": scores and goal events for that matchday's matches are
pre-generated deterministically (same style as data.py), but only
revealed incrementally as the simulated match clock advances.

Design choice: the simulated clock advances by explicit calls to
``tick()`` rather than real wall-clock time. This is what makes the
"live" experience -- the API layer calls ``tick()`` once per HTTP
request to ``/api/live``, so *polling the endpoint* is what advances
the match, exactly mirroring how a client repeatedly polling a real
live-score API would see the game progress. It also keeps this fully
deterministic and fast to test (no ``sleep()``/timing flakiness).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .models import Match, Season

MATCH_LENGTH_MINUTES = 90


@dataclass
class _LiveMatchPlan:
    """Pre-generated ground truth for one simulated live match.

    ``goal_events`` are already sorted by minute; ``final_home_score``/
    ``final_away_score`` are derived from them so the match always
    lands exactly on the pre-committed final score once minute reaches
    ``MATCH_LENGTH_MINUTES`` -- there's no separate "roll the final
    score" step that could disagree with the goal list.
    """

    match: Match
    goal_events: list[dict]
    final_home_score: int
    final_away_score: int


class LiveScoreSimulator:
    """Simulates one matchday's worth of matches going live and finishing.

    Usage: construct once per process (or per test) around a ``Season``
    and the matchday to simulate (defaults to the first unplayed
    matchday, i.e. the one immediately after the season's played
    cutoff). Call ``tick()`` to advance the shared match clock, then
    ``snapshot()`` to read the current state of every match on that
    matchday.
    """

    def __init__(
        self,
        season: Season,
        matchday: int | None = None,
        *,
        seed: int = 4242,
        minutes_per_tick: int = 6,
        max_goals_per_side: int = 5,
    ):
        self.season = season
        self.minutes_per_tick = minutes_per_tick
        self.minute = 0

        if matchday is None:
            unplayed = sorted({m.matchday for m in season.matches if not m.is_played})
            if not unplayed:
                raise ValueError(
                    "LiveScoreSimulator: season has no unplayed matchday to "
                    "simulate as live (every matchday is already played)"
                )
            matchday = unplayed[0]
        self.matchday = matchday

        live_matches = [m for m in season.matches if m.matchday == matchday]
        if not live_matches:
            raise ValueError(
                f"LiveScoreSimulator: matchday {matchday} has no matches"
            )

        rng = random.Random(seed)
        self._plans: list[_LiveMatchPlan] = []
        for match in live_matches:
            home_goals = self._sample_goal_count(rng, max_goals_per_side)
            away_goals = self._sample_goal_count(rng, max_goals_per_side)
            events = self._plan_goal_events(
                rng, match, home_goals, away_goals
            )
            self._plans.append(
                _LiveMatchPlan(
                    match=match,
                    goal_events=events,
                    final_home_score=home_goals,
                    final_away_score=away_goals,
                )
            )

    @staticmethod
    def _sample_goal_count(rng: random.Random, cap: int) -> int:
        count = 0
        p = 0.55
        while rng.random() < p and count < cap:
            count += 1
            p *= 0.5
        return count

    _SCORER_WEIGHT_BY_POSITION = {"FWD": 6, "MID": 3, "DEF": 1, "GK": 0}

    def _plan_goal_events(
        self, rng: random.Random, match: Match, home_goals: int, away_goals: int
    ) -> list[dict]:
        total = home_goals + away_goals
        if total == 0:
            return []
        minutes = sorted(rng.sample(range(1, MATCH_LENGTH_MINUTES + 1), k=total))
        sides = ["home"] * home_goals + ["away"] * away_goals
        rng.shuffle(sides)

        home_team = self.season.find_team(match.home_team)
        away_team = self.season.find_team(match.away_team)
        events = []
        for minute, side in zip(minutes, sides):
            team_name = match.home_team if side == "home" else match.away_team
            roster = (home_team.roster if side == "home" else away_team.roster) if (home_team and away_team) else ()
            scorer = self._pick_scorer(rng, roster)
            events.append({
                "team": team_name,
                "minute": minute,
                "side": side,
                "player": scorer,
            })
        events.sort(key=lambda e: e["minute"])
        return events

    @classmethod
    def _pick_scorer(cls, rng: random.Random, roster: tuple) -> str | None:
        if not roster:
            return None
        weights = [cls._SCORER_WEIGHT_BY_POSITION.get(p.position, 1) for p in roster]
        if sum(weights) == 0:
            weights = [1] * len(roster)
        return rng.choices(roster, weights=weights, k=1)[0].name

    @property
    def is_finished(self) -> bool:
        return self.minute >= MATCH_LENGTH_MINUTES

    def tick(self) -> None:
        """Advance the simulated clock by ``minutes_per_tick`` minutes,
        clamped at full time. Calling ``tick()`` after the match has
        finished is a no-op (the match simply stays at its final score).
        """
        if self.minute >= MATCH_LENGTH_MINUTES:
            return
        self.minute = min(self.minute + self.minutes_per_tick, MATCH_LENGTH_MINUTES)

    def snapshot(self) -> dict:
        """Return the current state of every match on the live matchday."""
        matches_out = []
        for plan in self._plans:
            events_so_far = [e for e in plan.goal_events if e["minute"] <= self.minute]
            home_score = sum(1 for e in events_so_far if e["side"] == "home")
            away_score = sum(1 for e in events_so_far if e["side"] == "away")
            if self.minute <= 0:
                status = "SCHEDULED"
            elif self.minute >= MATCH_LENGTH_MINUTES:
                status = "FT"
            else:
                status = "LIVE"
            matches_out.append({
                "id": plan.match.id,
                "matchday": plan.match.matchday,
                "date": plan.match.date,
                "kickoff": plan.match.kickoff,
                "home_team": plan.match.home_team,
                "away_team": plan.match.away_team,
                "home_score": home_score if self.minute > 0 else None,
                "away_score": away_score if self.minute > 0 else None,
                "minute": self.minute,
                "status": status,
                "scorers": [
                    {"team": e["team"], "player": e["player"], "minute": e["minute"]}
                    for e in events_so_far
                ],
            })
        return {
            "matchday": self.matchday,
            "clock_minute": self.minute,
            "finished": self.is_finished,
            "matches": matches_out,
        }

    def reset(self) -> None:
        """Rewind the clock to kickoff (minute 0) without re-planning
        goals -- used by tests and by an operator wanting to replay the
        same simulated matchday.
        """
        self.minute = 0
