"""Leaderboard API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /scores`` — submit a score
  (``{"player": str, "game": str, "score": number}``)
- ``GET /leaderboard`` — ranked listing, highest score first
  (``?game=<name>`` filters to one game, ``?top=N`` limits the rows)
- ``GET /leaderboard/<game>`` — ranked listing for one game
  (``?top=N``)
- ``GET /players/<player>/best`` — best score per game for one player
  (``?game=<name>`` for a single game)
- ``GET /games`` — list games that have scores

Ties are broken by earliest submission: the first player to reach a
score ranks above a later player with the same score. Ranks are
1-based and shared on ties are not used — each row gets its own rank
in submission order.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory score store."""
    app = Flask(__name__)
    app.config["SCORES"] = []
    app.config["SEQ"] = 0

    def _error(message, status):
        return jsonify({"error": message}), status

    def _parse_top():
        top = request.args.get("top")
        if top is None:
            return None, None
        try:
            n = int(top)
        except ValueError:
            return None, _error("top must be an integer", 400)
        if n < 1:
            return None, _error("top must be at least 1", 400)
        return n, None

    def _ranked(scores):
        ordered = sorted(scores, key=lambda s: (-s["score"], s["seq"]))
        return [
            {
                "rank": i + 1,
                "player": s["player"],
                "game": s["game"],
                "score": s["score"],
                "submitted_at": s["submitted_at"],
            }
            for i, s in enumerate(ordered)
        ]

    @app.post("/scores")
    def submit_score():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        player = data.get("player")
        if not isinstance(player, str) or not player.strip():
            return _error("player is required and must be a non-empty string", 400)
        game = data.get("game")
        if not isinstance(game, str) or not game.strip():
            return _error("game is required and must be a non-empty string", 400)
        score = data.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
        ):
            return _error("score is required and must be a number", 400)
        app.config["SEQ"] += 1
        entry = {
            "player": player.strip(),
            "game": game.strip(),
            "score": score,
            "seq": app.config["SEQ"],
            "submitted_at": app.config["SEQ"],
        }
        app.config["SCORES"].append(entry)
        return jsonify(
            {
                "player": entry["player"],
                "game": entry["game"],
                "score": entry["score"],
                "submitted_at": entry["submitted_at"],
            }
        ), 201

    @app.get("/leaderboard")
    def leaderboard():
        top, err = _parse_top()
        if err is not None:
            return err
        game = request.args.get("game")
        scores = app.config["SCORES"]
        if game:
            scores = [s for s in scores if s["game"] == game]
        ranked = _ranked(scores)
        return jsonify(ranked[:top] if top else ranked)

    @app.get("/leaderboard/<path:game>")
    def game_leaderboard(game):
        top, err = _parse_top()
        if err is not None:
            return err
        scores = [s for s in app.config["SCORES"] if s["game"] == game]
        if not scores:
            return _error(f"no scores for game: {game}", 404)
        ranked = _ranked(scores)
        return jsonify(ranked[:top] if top else ranked)

    @app.get("/players/<path:player>/best")
    def player_best(player):
        game = request.args.get("game")
        scores = [s for s in app.config["SCORES"] if s["player"] == player]
        if game:
            scores = [s for s in scores if s["game"] == game]
        if not scores:
            return _error(f"no scores for player: {player}", 404)
        best = {}
        for s in scores:
            current = best.get(s["game"])
            if current is None or s["score"] > current["score"]:
                best[s["game"]] = s
        return jsonify(
            {
                "player": player,
                "best": [
                    {"game": g, "score": s["score"]}
                    for g, s in sorted(best.items())
                ],
            }
        )

    @app.get("/games")
    def list_games():
        return jsonify(sorted({s["game"] for s in app.config["SCORES"]}))

    return app
