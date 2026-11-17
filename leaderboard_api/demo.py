"""Demo the leaderboard_api: submit scores, rank, ties, top-N, bests."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from leaderboard_api.app import create_app

client = create_app().test_client()

submissions = [
    ("Ayesha", "snake", 1500),
    ("Bilal", "snake", 2200),
    ("Ayesha", "snake", 2200),   # ties Bilal, but submitted later
    ("Chen", "snake", 900),
    ("Bilal", "tetris", 3100),
    ("Ayesha", "tetris", 2750),
]
for player, game, score in submissions:
    resp = client.post("/scores", json={
        "player": player, "game": game, "score": score})
    entry = resp.get_json()
    print(f"Submitted: {entry['player']} scored {entry['score']} in "
          f"{entry['game']} (#{entry['submitted_at']}) -> {resp.status_code}")

snake = client.get("/leaderboard/snake").get_json()
print("Snake leaderboard (ties broken by earliest submission):")
for row in snake:
    print(f"  #{row['rank']} {row['player']}: {row['score']}")

top2 = client.get("/leaderboard?top=2").get_json()
print(f"Top 2 overall: {[(r['player'], r['game'], r['score']) for r in top2]}")

best = client.get("/players/Ayesha/best").get_json()
print(f"Ayesha's best per game: {best['best']}")

games = client.get("/games").get_json()
print(f"Games tracked: {games}")

missing = client.get("/leaderboard/chess")
print(f"Unknown game -> {missing.status_code}: {missing.get_json()['error']}")
bad = client.post("/scores", json={"player": "Dan", "game": "snake"})
print(f"Missing score -> {bad.status_code}: {bad.get_json()['error']}")
