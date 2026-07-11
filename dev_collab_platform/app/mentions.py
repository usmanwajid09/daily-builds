"""Parses @email mentions out of free-text comment bodies.

This app has no separate @username handle -- a person's identity is
their email -- so a mention looks like "cc @alice@example.com please
review". Extraction is deliberately permissive about what counts as a
candidate mention (anything shaped like an email after an @) and the
caller (comment_routes.create_comment) is responsible for filtering
that down to actual workspace members; an @mention of someone who
isn't in the workspace, or isn't a real account at all, is just
ignored rather than erroring the whole comment -- consistent with how
most chat/PM tools treat mentions of non-existent handles.
"""
import re

# A mention is "@" immediately followed by an email-shaped token. Stops
# at whitespace or common trailing punctuation a sentence would end a
# mention with (",", ".", "!", "?", ")", ";", ":") so "@a@b.com," in a
# sentence doesn't capture the trailing comma as part of the address.
_MENTION_RE = re.compile(r"@([^\s@,.;:!?)]+@[^\s@,.;:!?)]+\.[^\s@,.;:!?)]+)")


def extract_mentioned_emails(body: str) -> list:
    """Returns the sorted, de-duplicated, lower-cased list of email-like
    tokens mentioned in `body` via @email syntax. Does not validate that
    they're real accounts or workspace members -- that's the caller's job."""
    if not body:
        return []
    return sorted({m.lower() for m in _MENTION_RE.findall(body)})
