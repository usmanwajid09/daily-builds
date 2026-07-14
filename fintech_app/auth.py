"""Password hashing + JWT issue/decode for the fintech demo app.

Same shape as lms_platform/auth.py, dev_collab_platform/app/auth.py,
and saas_starter's auth module: bcrypt for password storage, HS256
JWTs. Per the dev_collab hotfix (PR #11), PyJWT >= ~2.4 requires the
`sub` claim to be a string per RFC 7519 -- the user id is always
encoded as `str(user_id)` and cast back to `int` on decode, never a
raw int.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import bcrypt
import jwt

JWT_ALG = "HS256"
JWT_TTL_SECONDS = 24 * 60 * 60

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised for any authentication failure (bad token, expired, etc)."""


@dataclass(frozen=True)
class Claims:
    user_id: int
    email: str


def is_valid_email(email: str) -> bool:
    return bool(email) and bool(EMAIL_RE.match(email))


def hash_password(password: str) -> str:
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # malformed hash, wrong types, etc. -- never let this crash the request
        return False


def issue_token(jwt_secret: str, user_id: int, email: str) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + JWT_TTL_SECONDS,
    }
    return jwt.encode(payload, jwt_secret, algorithm=JWT_ALG)


def decode_token(jwt_secret: str, token: str) -> Claims:
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("invalid token") from exc
    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthError("invalid token subject") from exc
    return Claims(user_id=user_id, email=payload.get("email", ""))
