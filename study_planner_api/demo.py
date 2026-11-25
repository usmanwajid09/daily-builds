"""Demo: create subjects, log study sessions, show per-subject totals and the daily goal check."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from study_planner_api.app import create_app


def main():
    client = create_app().test_client()

    math = client.post("/subjects", json={"name": "Maths", "daily_goal_minutes": 60}).get_json()
    physics = client.post("/subjects", json={"name": "Physics", "daily_goal_minutes": 45}).get_json()
    print(f"Created subjects: {math}, {physics}")

    for session in [
        {"subject_id": math["id"], "date": "2026-11-25", "minutes": 40, "topic": "Algebra"},
        {"subject_id": math["id"], "date": "2026-11-25", "minutes": 30, "topic": "Calculus"},
        {"subject_id": physics["id"], "date": "2026-11-25", "minutes": 50, "topic": "Mechanics"},
        {"subject_id": math["id"], "date": "2026-11-24", "minutes": 60, "topic": "Geometry"},
    ]:
        created = client.post("/sessions", json=session).get_json()
        print(f"Logged session: {created}")

    totals = client.get(f"/subjects/{math['id']}/totals").get_json()
    print(f"Maths totals: {totals}")

    goals = client.get("/goals/daily?date=2026-11-25").get_json()
    print(f"Daily goals 2026-11-25: {goals}")
    print(f"Headline: {goals['total_minutes']} minutes studied on 2026-11-25, "
          f"{goals['goals_met']}/{goals['goals_total']} goals met")

    bad = client.post("/sessions", json={"subject_id": math["id"], "minutes": -10})
    print(f"Negative minutes rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
