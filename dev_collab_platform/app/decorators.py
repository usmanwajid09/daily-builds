"""Flask decorator that authenticates a request via a Bearer JWT and
attaches the verified claims (user_id, workspace_id, role) to flask.g."""
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
        g.user_id = int(claims["sub"])  # "sub" is a string on the wire (RFC 7519)
        g.workspace_id = claims["workspace_id"]
        g.role = claims["role"]
        return fn(*args, **kwargs)

    return wrapper


def require_role(*roles):
    """Stack under @require_auth. Rejects if g.role isn't one of `roles`."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if g.role not in roles:
                return jsonify(error=f"requires role in {roles}"), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator
