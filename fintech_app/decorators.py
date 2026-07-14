"""Flask decorator for Bearer-JWT authentication."""
from functools import wraps

from flask import current_app, g, jsonify, request

from . import auth


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify(error="missing bearer token"), 401
        token = header[len("Bearer "):].strip()
        try:
            claims = auth.decode_token(current_app.config["JWT_SECRET"], token)
        except auth.AuthError as exc:
            return jsonify(error=str(exc)), 401
        g.user_id = claims.user_id
        g.email = claims.email
        return fn(*args, **kwargs)

    return wrapper
