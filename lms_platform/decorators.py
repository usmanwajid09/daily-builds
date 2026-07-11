"""Flask decorators for Bearer-JWT authentication and role gating."""
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
        g.role = claims.role
        return fn(*args, **kwargs)

    return wrapper


def require_role(*roles):
    """Stack under @require_auth. Rejects with 403 unless g.role is in roles."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if g.role not in roles:
                return jsonify(error=f"requires role in {list(roles)}"), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator
