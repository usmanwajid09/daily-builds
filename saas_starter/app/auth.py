"""Password hashing and JWT issuing/verification for the SaaS starter.

JWT claims:
    sub        -- user id
    tenant_id  -- the tenant this token is scoped to (its "active org")
    role       -- the user's role within that tenant, at issue time
    exp        -- expiry (UTC)

A user with memberships in multiple tenants gets a *different* token per
tenant (issued at login, or via /api/auth/switch-tenant). Routes never
trust a tenant_id supplied by the client in the request body/query
string -- only the one embedded in the verified token -- so there is no
way to read another tenant's data by just changing an id in the URL.
"""
import re
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

DEFAULT_JWT_ALGORITHM = "HS256"
DEFAULT_TOKEN_TTL_MINUTES = 60

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised for any auth-related failure (bad credentials, bad token)."""


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


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "org"


def issue_token(secret: str, user_id: int, tenant_id: int, role: str,
                 ttl_minutes: int = DEFAULT_TOKEN_TTL_MINUTES,
                 algorithm: str = DEFAULT_JWT_ALGORITHM) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_token(secret: str, token: str, algorithm: str = DEFAULT_JWT_ALGORITHM) -> dict:
    try:
        return jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("invalid token") from exc
