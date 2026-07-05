import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from app.auth import (  # noqa: E402
    AuthError,
    decode_token,
    hash_password,
    is_valid_email,
    issue_token,
    slugify,
    verify_password,
)


def test_hash_and_verify_roundtrip():
    h = hash_password("correct-horse-battery")
    assert verify_password("correct-horse-battery", h)
    assert not verify_password("wrong-password", h)


def test_hash_password_rejects_short_password():
    with pytest.raises(ValueError):
        hash_password("short")


def test_verify_password_handles_garbage_hash_gracefully():
    assert not verify_password("anything", "not-a-real-bcrypt-hash")


def test_is_valid_email():
    assert is_valid_email("a@example.com")
    assert not is_valid_email("not-an-email")
    assert not is_valid_email("")
    assert not is_valid_email(None)


def test_slugify():
    assert slugify("Acme Corp!") == "acme-corp"
    assert slugify("   ") == "org"
    assert slugify("Already-Slug") == "already-slug"


def test_issue_and_decode_token_roundtrip():
    token = issue_token("secret", user_id=1, tenant_id=2, role="owner")
    claims = decode_token("secret", token)
    assert claims["sub"] == 1
    assert claims["tenant_id"] == 2
    assert claims["role"] == "owner"


def test_decode_token_rejects_wrong_secret():
    token = issue_token("secret", user_id=1, tenant_id=2, role="owner")
    with pytest.raises(AuthError):
        decode_token("wrong-secret", token)


def test_decode_token_rejects_expired_token():
    token = issue_token("secret", user_id=1, tenant_id=2, role="owner", ttl_minutes=-1)
    with pytest.raises(AuthError):
        decode_token("secret", token)
