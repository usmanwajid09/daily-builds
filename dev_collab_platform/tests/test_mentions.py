import pytest

from app.mentions import extract_mentioned_emails


@pytest.mark.parametrize("body,expected", [
    ("no mentions here", []),
    ("cc @alice@example.com please review", ["alice@example.com"]),
    ("cc @Alice@Example.com and @ALICE@example.com", ["alice@example.com"]),  # case-insensitive dedup
    ("@a@b.com and @c@d.org both", ["a@b.com", "c@d.org"]),
    ("(@a@b.com)", ["a@b.com"]),  # trailing paren not captured
    ("hi @a@b.com, how are you?", ["a@b.com"]),  # trailing comma not captured
    ("sentence ending in a mention @a@b.com.", ["a@b.com"]),  # trailing period not captured
    ("not an email: @plainword", []),
    ("", []),
])
def test_extract_mentioned_emails(body, expected):
    assert extract_mentioned_emails(body) == expected


def test_extract_mentioned_emails_none_input():
    assert extract_mentioned_emails(None) == []


def test_extract_mentioned_emails_sorted_output():
    result = extract_mentioned_emails("@zed@example.com and @amy@example.com")
    assert result == sorted(result)
