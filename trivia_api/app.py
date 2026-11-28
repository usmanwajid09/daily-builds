"""Flask trivia API with an in-memory question store and player scores."""

from flask import Flask, jsonify, request

DIFFICULTIES = ("easy", "medium", "hard")

SEED_QUESTIONS = [
    {
        "question": "What is the chemical symbol for water?",
        "options": ["CO2", "H2O", "O2", "NaCl"],
        "answer_index": 1,
        "category": "science",
        "difficulty": "easy",
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "options": ["Venus", "Jupiter", "Mars", "Saturn"],
        "answer_index": 2,
        "category": "science",
        "difficulty": "easy",
    },
    {
        "question": "What is the powerhouse of the cell?",
        "options": ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"],
        "answer_index": 2,
        "category": "science",
        "difficulty": "medium",
    },
    {
        "question": "What is the speed of light in a vacuum, approximately?",
        "options": ["150,000 km/s", "300,000 km/s", "450,000 km/s", "600,000 km/s"],
        "answer_index": 1,
        "category": "science",
        "difficulty": "hard",
    },
    {
        "question": "In which year did World War II end?",
        "options": ["1943", "1944", "1945", "1946"],
        "answer_index": 2,
        "category": "history",
        "difficulty": "easy",
    },
    {
        "question": "Who was the first President of the United States?",
        "options": ["Abraham Lincoln", "Thomas Jefferson", "George Washington", "John Adams"],
        "answer_index": 2,
        "category": "history",
        "difficulty": "easy",
    },
    {
        "question": "Which ancient civilization built the pyramids of Giza?",
        "options": ["Romans", "Greeks", "Egyptians", "Aztecs"],
        "answer_index": 2,
        "category": "history",
        "difficulty": "medium",
    },
    {
        "question": "The Treaty of Versailles was signed in which year?",
        "options": ["1917", "1918", "1919", "1920"],
        "answer_index": 2,
        "category": "history",
        "difficulty": "hard",
    },
    {
        "question": "What is the capital of Australia?",
        "options": ["Sydney", "Melbourne", "Canberra", "Perth"],
        "answer_index": 2,
        "category": "geography",
        "difficulty": "easy",
    },
    {
        "question": "Which is the longest river in the world?",
        "options": ["Amazon", "Yangtze", "Mississippi", "Nile"],
        "answer_index": 3,
        "category": "geography",
        "difficulty": "medium",
    },
    {
        "question": "Which country has the largest population?",
        "options": ["United States", "Indonesia", "India", "China"],
        "answer_index": 2,
        "category": "geography",
        "difficulty": "hard",
    },
    {
        "question": "How many players are on a football (soccer) team on the pitch?",
        "options": ["9", "10", "11", "12"],
        "answer_index": 2,
        "category": "sports",
        "difficulty": "easy",
    },
    {
        "question": "In which sport would you perform a slam dunk?",
        "options": ["Volleyball", "Basketball", "Tennis", "Badminton"],
        "answer_index": 1,
        "category": "sports",
        "difficulty": "easy",
    },
    {
        "question": "How often are the Olympic Games held?",
        "options": ["Every 2 years", "Every 3 years", "Every 4 years", "Every 5 years"],
        "answer_index": 2,
        "category": "sports",
        "difficulty": "medium",
    },
    {
        "question": "What does HTTP stand for?",
        "options": [
            "HyperText Transfer Protocol",
            "HighText Transfer Protocol",
            "HyperText Transmission Process",
            "Host Transfer Text Protocol",
        ],
        "answer_index": 0,
        "category": "technology",
        "difficulty": "easy",
    },
    {
        "question": "Which language is primarily used to style web pages?",
        "options": ["HTML", "Python", "CSS", "SQL"],
        "answer_index": 2,
        "category": "technology",
        "difficulty": "medium",
    },
]


def _public(question):
    """Question view without the answer (safe to show to players)."""
    return {key: value for key, value in question.items() if key != "answer_index"}


def create_app():
    """Return a fresh Flask app with a seeded in-memory trivia store."""
    app = Flask(__name__)

    questions = []
    for index, seed in enumerate(SEED_QUESTIONS, start=1):
        questions.append({"id": index, **seed})
    next_id = [len(questions) + 1]
    players = {}  # name -> {"score": int, "answered": int}

    def _find(question_id):
        for question in questions:
            if question["id"] == question_id:
                return question
        return None

    def _record(player):
        record = players[player]
        accuracy = round(record["score"] / record["answered"], 2) if record["answered"] else 0.0
        return {
            "player": player,
            "score": record["score"],
            "answered": record["answered"],
            "accuracy": accuracy,
        }

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": str(error.description or "Bad request")}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": str(error.description or "Not found")}), 404

    @app.get("/questions")
    def list_questions():
        category = request.args.get("category")
        difficulty = request.args.get("difficulty")
        if difficulty is not None and difficulty not in DIFFICULTIES:
            return jsonify({"error": f"Invalid difficulty: {difficulty!r}. "
                            "Must be one of: easy, medium, hard."}), 400
        result = questions
        if category is not None:
            result = [q for q in result if q["category"] == category]
        if difficulty is not None:
            result = [q for q in result if q["difficulty"] == difficulty]
        return jsonify([_public(q) for q in result])

    @app.get("/questions/<int:question_id>")
    def get_question(question_id):
        question = _find(question_id)
        if question is None:
            return jsonify({"error": f"Question {question_id} not found."}), 404
        return jsonify(_public(question))

    @app.post("/questions")
    def add_question():
        data = request.get_json(silent=True) or {}
        text = data.get("question")
        options = data.get("options")
        answer_index = data.get("answer_index")
        category = data.get("category")
        difficulty = data.get("difficulty")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Field 'question' must be a non-empty string."}), 400
        if (not isinstance(options, list) or len(options) < 2
                or not all(isinstance(o, str) and o.strip() for o in options)):
            return jsonify({"error": "Field 'options' must be a list of at least 2 strings."}), 400
        if (not isinstance(answer_index, int) or isinstance(answer_index, bool)
                or not 0 <= answer_index < len(options)):
            return jsonify({"error": "Field 'answer_index' must be an integer within the options range."}), 400
        if not isinstance(category, str) or not category.strip():
            return jsonify({"error": "Field 'category' must be a non-empty string."}), 400
        if difficulty not in DIFFICULTIES:
            return jsonify({"error": "Field 'difficulty' must be one of: easy, medium, hard."}), 400
        question = {
            "id": next_id[0],
            "question": text,
            "options": options,
            "answer_index": answer_index,
            "category": category,
            "difficulty": difficulty,
        }
        next_id[0] += 1
        questions.append(question)
        return jsonify(question), 201

    @app.post("/answer")
    def answer():
        data = request.get_json(silent=True) or {}
        player = data.get("player")
        question_id = data.get("question_id")
        answer_index = data.get("answer_index")
        if not isinstance(player, str) or not player.strip():
            return jsonify({"error": "Field 'player' is required."}), 404
        if (not isinstance(question_id, int) or isinstance(question_id, bool)):
            return jsonify({"error": "Field 'question_id' must be an integer."}), 400
        question = _find(question_id)
        if question is None:
            return jsonify({"error": f"Question {question_id} not found."}), 404
        if (not isinstance(answer_index, int) or isinstance(answer_index, bool)
                or not 0 <= answer_index < len(question["options"])):
            return jsonify({"error": "Field 'answer_index' is out of range for this question."}), 400
        record = players.setdefault(player, {"score": 0, "answered": 0})
        record["answered"] += 1
        correct = answer_index == question["answer_index"]
        if correct:
            record["score"] += 1
        return jsonify({
            "correct": correct,
            "correct_index": question["answer_index"],
            "player": player,
            "score": record["score"],
            "answered": record["answered"],
        })

    @app.get("/scores")
    def leaderboard():
        board = sorted(
            (_record(name) for name in players),
            key=lambda r: (-r["score"], r["player"]),
        )
        return jsonify(board)

    @app.get("/scores/<player>")
    def player_score(player):
        if player not in players:
            return jsonify({"error": f"Player {player!r} has not answered any questions."}), 404
        return jsonify(_record(player))

    return app
