"""Rule-based transaction categorization engine.

Pure and stateless: given a merchant name, a free-text description, and
a signed amount, decide which spending category the transaction belongs
to. No network calls, no ML model to train/ship -- a simple, auditable
keyword-rule engine, which is the right tool here since real bank
transaction descriptions are short and merchant-code-like, and rule
matches are easy for a user to understand ("why was this categorized as
'dining'?" -> "it matched the word 'cafe'").

Matching strategy:
  1. If the amount is positive (money IN) and no stronger rule matches
     first, it defaults to "income" -- but explicit rules (e.g. a refund
     from a known merchant) still take priority over the sign-based
     default, so a merchant match always wins over the amount-sign guess.
  2. Rules are grouped by category and checked in a fixed priority order
     (CATEGORY_PRIORITY) so that a more specific category (e.g.
     "subscription") is chosen over a broader one (e.g. "shopping") when
     a description could plausibly match both.
  3. Matching is case-insensitive substring matching against a combined
     "merchant description" text blob, so partial merchant names (as
     real bank feeds often truncate/abbreviate them) still match.
  4. If nothing matches, falls back to "income" (positive amount) or
     "other" (negative amount).
"""
from __future__ import annotations

# Ordered most-specific-first: earlier categories win ties when a blob
# matches keywords from more than one category.
CATEGORY_PRIORITY = [
    "income",
    "transfer",
    "housing",
    "subscription",
    "utilities",
    "health",
    "transport",
    "groceries",
    "dining",
    "entertainment",
    "fees",
    "shopping",
]

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "income": [
        "payroll", "salary", "direct deposit", "employer", "paycheck",
        "dividend", "interest payment", "tax refund", "reimbursement",
    ],
    "transfer": [
        "transfer to", "transfer from", "zelle", "venmo", "paypal transfer",
        "ach transfer", "wire transfer", "account transfer",
    ],
    "housing": [
        "rent payment", "mortgage", "property management", "landlord",
        "homeowners insurance", "hoa fee",
    ],
    "subscription": [
        "netflix", "spotify", "hulu", "disney+", "apple.com/bill",
        "subscription", "amazon prime", "youtube premium", "gym membership",
        "membership fee",
    ],
    "utilities": [
        "electric", "power company", "water utility", "gas company",
        "internet service", "broadband", "phone bill", "wireless carrier",
        "utility",
    ],
    "health": [
        "pharmacy", "cvs", "walgreens", "urgent care", "medical center",
        "clinic", "hospital", "dental", "doctor", "insurance premium",
    ],
    "transport": [
        "uber", "lyft", "gas station", "fuel", "parking", "transit",
        "metro card", "toll", "airline", "car rental",
    ],
    "groceries": [
        "grocery", "supermarket", "whole foods", "trader joe", "safeway",
        "kroger", "market basket", "farmers market",
    ],
    "dining": [
        "restaurant", "cafe", "coffee", "starbucks", "diner", "bistro",
        "pizzeria", "food truck", "bar & grill", "takeout",
    ],
    "entertainment": [
        "cinema", "movie theater", "concert", "ticketmaster", "steam games",
        "playstation store", "arcade", "bowling",
    ],
    "fees": [
        "overdraft fee", "late fee", "service charge", "atm fee",
        "maintenance fee", "nsf fee", "annual fee",
    ],
    "shopping": [
        "amazon.com", "target", "walmart", "best buy", "department store",
        "online store", "mall", "clothing",
    ],
}

FALLBACK_POSITIVE = "income"
FALLBACK_NEGATIVE = "other"

ALL_CATEGORIES = CATEGORY_PRIORITY + [FALLBACK_NEGATIVE]


def categorize(merchant: str, description: str, amount: float) -> str:
    """Return the best-matching category for a transaction.

    `amount` follows the app-wide sign convention: positive = money in,
    negative = money out. It is only used as a fallback / tie-breaker,
    never to override an explicit keyword match (a "refund" description
    on a negative amount should still be categorized by its words, not
    blindly treated as an outflow-only category).
    """
    blob = f"{merchant} {description}".lower()

    for category in CATEGORY_PRIORITY:
        for keyword in CATEGORY_KEYWORDS[category]:
            if keyword in blob:
                return category

    return FALLBACK_POSITIVE if amount > 0 else FALLBACK_NEGATIVE


def explain(merchant: str, description: str, amount: float) -> dict:
    """Like categorize(), but also returns which keyword/category matched
    (or None if it fell back to the sign-based default) -- useful for a
    "why was this categorized this way" UI and for tests."""
    blob = f"{merchant} {description}".lower()

    for category in CATEGORY_PRIORITY:
        for keyword in CATEGORY_KEYWORDS[category]:
            if keyword in blob:
                return {"category": category, "matched_keyword": keyword, "used_fallback": False}

    category = FALLBACK_POSITIVE if amount > 0 else FALLBACK_NEGATIVE
    return {"category": category, "matched_keyword": None, "used_fallback": True}
