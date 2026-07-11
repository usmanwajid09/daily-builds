
import pytest

from app import auth


def test_password_hash_roundtrip():
    h = auth.hash_password("correct horse battery")
    assert auth.verify_password("correct horse battery", h)
    assert not auth.verify_password("wrong password", h)


def test_password_too_short_rejected():
    with pytest.raises(ValueError):
        auth.hash_password("short")


def test_verify_password_never_raises_on_malformed_hash():
    assert auth.verify_password("anything", "not-a-real-bcrypt-hash") is False
    assert auth.verify_password("anything", "") is False


@pytest.mark.parametrize("email,expected", [
    ("a@example.com", True),
    ("a.b+tag@example.co.uk", True),
    ("not-an-email", False),
    ("missing-at.example.com", False),
    ("", False),
    (None, False),
])
def test_is_valid_email(email, expected):
    assert auth.is_valid_email(email) is expected


@pytest.mark.parametrize("name,expected", [
    ("Acme Corp", "acme-corp"),
    ("  spaced   out  ", "spaced-out"),
    ("!!!", "workspace"),
    ("Already-Slug", "already-slug"),
])
def test_slugify(name, expected):
    assert auth.slugify(name) == expected


def test_issue_and_decode_token_roundtrip():
    token = auth.issue_token("secret", user_id=7, workspace_id=3, role="admin")
    claims = auth.decode_token("secret", token)
    assert claims["sub"] == "7"  # RFC 7519: "sub" is a StringOrURI on the wire
    assert claims["workspace_id"] == 3
    assert claims["role"] == "admin"


def test_issue_token_encodes_sub_as_a_string_not_an_int():
    """Regression test: RFC 7519 says "sub" should be a StringOrURI.
    Older PyJWT (e.g. 2.3.x) never checked this on decode, so encoding
    "sub" as a raw int worked there but broke on newer PyJWT releases
    that validate it and raise InvalidSubjectError("Subject must be a
    string") -- surfaced through our code as a generic, misleading
    "invalid token" 401 on every single authenticated request. Assert
    the wire format directly (not just that decode succeeds) so this
    can't silently regress back to an int."""
    token = auth.issue_token("secret", user_id=42, workspace_id=1, role="owner")
    header, payload_b64, _sig = token.split(".")
    import base64
    import json
    padded = payload_b64 + "=" * (-len(payload_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded))
    assert payload["sub"] == "42"
    assert isinstance(payload["sub"], str)


def test_decode_token_wrong_secret_fails():
    token = auth.issue_token("secret", user_id=7, workspace_id=3, role="admin")
    with pytest.raises(auth.AuthError):
        auth.decode_token("wrong-secret", token)


def test_decode_expired_token_fails():
    token = auth.issue_token("secret", user_id=1, workspace_id=1, role="owner", ttl_minutes=-1)
    with pytest.raises(auth.AuthError):
        auth.decode_token("secret", token)


def test_decode_garbage_token_fails():
    with pytest.raises(auth.AuthError):
        auth.decode_token("secret", "not.a.jwt")
