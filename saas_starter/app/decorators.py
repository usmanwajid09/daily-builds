"""Auth decorators shared by protected routes."""
import functools

from flask import current_app, g, jsonify, request

from . import db
from .auth import AuthError, decode_token


def _extract_bearer_token() -> str | None:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header[len("Bearer "):].strip() or None


def jwt_required(fn):
    """Decode the bearer token and attach g.current_user_id / g.tenant_id /
    g.role. Re-checks that the membership still exists (e.g. hasn't been
    removed since the token was issued) so a revoked user can't keep using
    a still-valid-looking token for the rest of its TTL.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = _extract_bearer_token()
        if not token:
            return jsonify(error="missing bearer token"), 401
        try:
            claims = decode_token(current_app.config["JWT_SECRET"], token)
        except AuthError as exc:
            return jsonify(error=str(exc)), 401

        user_id = claims["sub"]
        tenant_id = claims["tenant_id"]

        with db.connect(current_app.config["DB_PATH"]) as conn:
            membership = db.get_membership(conn, user_id, tenant_id)
        if membership is None:
            return jsonify(error="membership revoked"), 401

        g.current_user_id = user_id
        g.tenant_id = tenant_id
        g.role = membership["role"]
        return fn(*args, **kwargs)

    return wrapper


def role_required(*allowed_roles):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if g.get("role") not in allowed_roles:
                return jsonify(error="insufficient role"), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator
