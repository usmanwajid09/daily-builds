"""Password utility API built with Flask.

Endpoints:

- ``POST /generate`` — random password. Options: ``length`` (4–128,
  default 16), ``symbols``, ``digits``, ``uppercase``, ``lowercase``
  (booleans, all default true). At least one character class must stay
  enabled, and the result contains at least one character from every
  enabled class.
- ``POST /strength`` — ``{"password": str}`` returns a score from 0–5,
  a label (Weak / Fair / Good / Strong), and a list of feedback tips.
- ``GET /`` — service info

Generation uses :mod:`secrets` (cryptographically strong randomness),
and no password is ever stored.
"""
import secrets
import string

from flask import Flask, jsonify, request

MIN_LENGTH = 4
MAX_LENGTH = 128
DEFAULT_LENGTH = 16

_CLASS_POOLS = {
    "lowercase": string.ascii_lowercase,
    "uppercase": string.ascii_uppercase,
    "digits": string.digits,
    "symbols": string.punctuation,
}


def generate_password(length=DEFAULT_LENGTH, **enabled):
    """Generate a random password of the given length.

    ``enabled`` maps each character class name to a bool. At least one
    class must be enabled, and the password always contains at least
    one character from every enabled class.
    """
    if isinstance(length, bool) or not isinstance(length, int):
        raise ValueError("length must be an integer")
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        raise ValueError(f"length must be between {MIN_LENGTH} and {MAX_LENGTH}")
    active = [name for name, on in enabled.items() if on]
    if not active:
        raise ValueError("at least one character class must be enabled")
    if length < len(active):
        raise ValueError(
            f"length must be at least {len(active)} to include every enabled class"
        )
    rng = secrets.SystemRandom()
    alphabet = "".join(_CLASS_POOLS[name] for name in active)
    chars = [rng.choice(_CLASS_POOLS[name]) for name in active]
    chars += [rng.choice(alphabet) for _ in range(length - len(active))]
    rng.shuffle(chars)
    return "".join(chars)


def check_strength(password):
    """Score a password 0–5 and return feedback on how to improve it.

    One point each for: length >= 8, length >= 12, mixed case, a digit,
    and a symbol.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    feedback = []
    score = 0
    if len(password) >= 8:
        score += 1
    else:
        feedback.append("use at least 8 characters")
    if len(password) >= 12:
        score += 1
    else:
        feedback.append("use at least 12 characters for a stronger password")
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    if has_lower and has_upper:
        score += 1
    else:
        feedback.append("mix uppercase and lowercase letters")
    if any(c.isdigit() for c in password):
        score += 1
    else:
        feedback.append("add a digit")
    if any(c in string.punctuation for c in password):
        score += 1
    else:
        feedback.append("add a symbol (e.g. !, @, #)")
    if score <= 1:
        label = "Weak"
    elif score == 2:
        label = "Fair"
    elif score == 3:
        label = "Good"
    else:
        label = "Strong"
    return {"score": score, "label": label, "feedback": feedback}


def create_app():
    """Return the password utility Flask app."""
    app = Flask(__name__)

    def _error(message, status):
        return jsonify({"error": message}), status

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "password_api",
                "usage": "POST /generate or POST /strength with a JSON body",
            }
        )

    @app.post("/generate")
    def generate():
        data = request.get_json(silent=True)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        options = {}
        for name in _CLASS_POOLS:
            value = data.get(name, True)
            if not isinstance(value, bool):
                return _error(f"{name} must be a boolean", 400)
            options[name] = value
        try:
            password = generate_password(data.get("length", DEFAULT_LENGTH), **options)
        except ValueError as exc:
            return _error(str(exc), 400)
        return jsonify({"password": password, "length": len(password)})

    @app.post("/strength")
    def strength():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        password = data.get("password")
        if not isinstance(password, str) or not password:
            return _error("password is required and must be a non-empty string", 400)
        return jsonify(check_strength(password))

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
