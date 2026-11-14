"""Flask currency converter API.

Converts between currencies using a small built-in rate table of
**demo rates** (units of each currency per 1 USD — not live market
rates, and documented as such in the README). Rates can be added or
updated at runtime; the store is in-memory and fresh for each app
instance. All responses are JSON. Endpoints:

- ``GET /currencies`` — list all currencies and their rates
- ``GET /rates/<code>`` — one currency's rate
- ``GET /convert?from=USD&to=PKR&amount=10`` — convert an amount
- ``POST /rates`` — add or update a rate (``{"code": "EUR",
  "rate": 0.92}``)
"""
from flask import Flask, jsonify, request

# Demo rates: units of the currency per 1 USD. NOT live market rates.
DEFAULT_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "PKR": 278.0,
    "INR": 83.5,
    "JPY": 149.0,
    "AED": 3.67,
    "CNY": 7.24,
    "CAD": 1.36,
    "AUD": 1.52,
}


def create_app():
    """Return a Flask app with a fresh copy of the demo rate table."""
    app = Flask(__name__)
    app.config["RATES"] = dict(DEFAULT_RATES)

    def _error(message, status):
        return jsonify({"error": message}), status

    def _parse_code(value):
        if not isinstance(value, str):
            return None
        code = value.strip().upper()
        if len(code) != 3 or not code.isalpha():
            return None
        return code

    def _parse_amount(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if value <= 0:
            return None
        return float(value)

    @app.get("/currencies")
    def list_currencies():
        rates = app.config["RATES"]
        return jsonify(
            [{"code": code, "rate_per_usd": rates[code]} for code in sorted(rates)]
        )

    @app.get("/rates/<code>")
    def get_rate(code):
        parsed = _parse_code(code)
        if parsed is None:
            return _error("currency code must be 3 letters, e.g. USD", 400)
        rate = app.config["RATES"].get(parsed)
        if rate is None:
            return _error(f"unknown currency {parsed}", 404)
        return jsonify({"code": parsed, "rate_per_usd": rate})

    @app.get("/convert")
    def convert():
        from_code = _parse_code(request.args.get("from"))
        to_code = _parse_code(request.args.get("to"))
        if from_code is None or to_code is None:
            return _error(
                "from and to are required and must be 3-letter currency codes", 400
            )
        raw_amount = request.args.get("amount")
        if raw_amount is None:
            return _error("amount is required", 400)
        try:
            amount = float(raw_amount)
        except ValueError:
            return _error("amount must be a number", 400)
        amount = _parse_amount(amount)
        if amount is None:
            return _error("amount must be a positive number", 400)
        rates = app.config["RATES"]
        if from_code not in rates:
            return _error(f"unknown currency {from_code}", 400)
        if to_code not in rates:
            return _error(f"unknown currency {to_code}", 400)
        rate = rates[to_code] / rates[from_code]
        return jsonify(
            {
                "from": from_code,
                "to": to_code,
                "amount": amount,
                "rate": rate,
                "result": amount * rate,
                "demo_rates": True,
            }
        )

    @app.post("/rates")
    def set_rate():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        code = _parse_code(data.get("code"))
        if code is None:
            return _error("code is required and must be a 3-letter currency code", 400)
        rate = data.get("rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
            return _error("rate must be a positive number", 400)
        is_new = code not in app.config["RATES"]
        app.config["RATES"][code] = float(rate)
        return jsonify({"code": code, "rate_per_usd": float(rate)}), (
            201 if is_new else 200
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
