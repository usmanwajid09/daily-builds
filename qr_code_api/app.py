"""Flask 2D barcode-style code API (no image dependencies).

Encodes text into a deterministic grid of 0/1 bits — the UTF-8 bytes of
the text, most-significant bit first, laid out row by row in a grid
16 bits wide, zero-padded at the end — and decodes such a grid back to
text. The same text always produces the same grid, so encode/decode
round-trips exactly. All responses are JSON. Endpoints:

- ``POST /encode`` — ``{"text": str}`` -> ``{"text", "rows", "cols",
  "grid", "ascii"}`` where ``ascii`` is a ``#``/space rendering
- ``POST /decode`` — ``{"grid": [[0|1, ...], ...]}`` -> ``{"text": str}``

The ``#``/space ASCII rendering is a human-readable picture of the
grid, not a scannable QR code; no imaging library is involved.
"""
from flask import Flask, jsonify, request

COLS = 16
MAX_TEXT_LENGTH = 256


def create_app():
    """Return a Flask app for the barcode-code endpoints."""
    app = Flask(__name__)

    def _error(message, status):
        return jsonify({"error": message}), status

    def _encode_bits(text):
        bits = []
        for byte in text.encode("utf-8"):
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
        return bits

    def _grid_to_ascii(grid):
        return "\n".join(
            "".join("#" if cell else " " for cell in row) for row in grid
        )

    @app.post("/encode")
    def encode():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        text = data.get("text")
        if not isinstance(text, str) or not text:
            return _error("text is required and must be a non-empty string", 400)
        if "\x00" in text:
            return _error("text must not contain NUL characters", 400)
        if len(text) > MAX_TEXT_LENGTH:
            return _error(
                f"text must be at most {MAX_TEXT_LENGTH} characters", 400
            )
        bits = _encode_bits(text)
        rows = (len(bits) + COLS - 1) // COLS
        bits.extend([0] * (rows * COLS - len(bits)))
        grid = [bits[i * COLS:(i + 1) * COLS] for i in range(rows)]
        return jsonify(
            {
                "text": text,
                "rows": rows,
                "cols": COLS,
                "grid": grid,
                "ascii": _grid_to_ascii(grid),
            }
        )

    @app.post("/decode")
    def decode():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        grid = data.get("grid")
        if not isinstance(grid, list) or not grid:
            return _error("grid is required and must be a non-empty list of rows", 400)
        width = None
        bits = []
        for row in grid:
            if not isinstance(row, list) or not row:
                return _error("each grid row must be a non-empty list", 400)
            if width is None:
                width = len(row)
            elif len(row) != width:
                return _error("all grid rows must have the same length", 400)
            for cell in row:
                if cell not in (0, 1) or isinstance(cell, bool):
                    return _error("grid cells must be 0 or 1", 400)
                bits.append(cell)
        raw = bytearray()
        for i in range(0, len(bits) - len(bits) % 8, 8):
            byte = 0
            for bit in bits[i:i + 8]:
                byte = (byte << 1) | bit
            raw.append(byte)
        while raw and raw[-1] == 0:
            raw.pop()
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            return _error("grid does not decode to valid UTF-8 text", 400)
        return jsonify({"text": text, "rows": len(grid), "cols": width})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
