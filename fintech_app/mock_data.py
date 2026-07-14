"""Seeded synthetic account + transaction generator.

This is the ONLY source of "bank" data in this app -- there is no real
bank integration, and per the arc's safety rule there never will be.
Everything here is deterministic given a seed, so the same user always
gets the same demo data (useful for tests and for a stable demo).

Generates 3 accounts (checking, savings, credit_card) per user and a
few months of transactions per account, including a handful of
recurring monthly items (rent, a subscription, a utility bill) planted
on a fixed day-of-month with near-constant amounts -- specifically so
milestone 2's recurring-bill detector has real signal to find.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

# (merchant, description, category-ish flavor, amount range) -- amount
# range excludes sign, applied by the caller based on transaction type.
DISCRETIONARY_SPENDS = [
    ("Whole Foods Market", "grocery purchase", (25, 140)),
    ("Trader Joe's", "grocery purchase", (15, 90)),
    ("Starbucks", "coffee", (4, 9)),
    ("Corner Bistro", "restaurant", (18, 65)),
    ("Uber", "rideshare trip", (8, 35)),
    ("Shell Gas Station", "fuel", (25, 60)),
    ("Amazon.com", "online store purchase", (12, 180)),
    ("Target", "department store purchase", (10, 120)),
    ("AMC Cinema", "movie theater tickets", (12, 40)),
    ("CVS Pharmacy", "pharmacy purchase", (8, 55)),
]

RECURRING_MONTHLY = [
    # (merchant, description, day_of_month, amount) -- amount is negative
    # (money out); a small amount of jitter is applied per month so it's
    # realistic but still clearly "the same bill" for detection purposes.
    ("Skyline Property Management", "rent payment", 1, 1450.00),
    ("Netflix", "monthly subscription", 5, 15.49),
    ("Citywide Power Company", "electric utility bill", 12, 68.00),
    ("Metro Wireless", "phone bill", 18, 55.00),
]

PAYCHECK = ("Acme Corp Payroll", "direct deposit payroll", 2600.00)


def _daterange_months_back(months: int, today: date) -> list[date]:
    """All calendar days from `months` months ago through today."""
    start = date(today.year, today.month, 1)
    for _ in range(months - 1):
        start = (start.replace(day=1) - timedelta(days=1)).replace(day=1)
    days = []
    d = start
    while d <= today:
        days.append(d)
        d += timedelta(days=1)
    return days


def generate_mock_accounts() -> list[dict]:
    """Fixed set of 3 demo accounts -- name/type/institution only, no
    balances (balances are always derived from transactions)."""
    return [
        {"name": "Everyday Checking", "account_type": "checking", "institution_name": "Demo Bank"},
        {"name": "Rainy Day Savings", "account_type": "savings", "institution_name": "Demo Bank"},
        {"name": "Demo Rewards Card", "account_type": "credit_card", "institution_name": "Demo Bank"},
    ]


def generate_mock_transactions(account_type: str, seed: int, months: int = 3,
                                today: date | None = None) -> list[dict]:
    """Deterministic synthetic transactions for one account, given a seed.

    Returns a list of dicts with keys: merchant, description, amount,
    posted_at (ISO date string). Category is intentionally NOT set here
    -- the caller runs categorization.categorize() on each row, exactly
    as it would for a manually-entered transaction, so the demo data
    exercises the same code path as real usage.
    """
    today = today or date.today()
    rng = random.Random(seed)
    days = _daterange_months_back(months, today)
    transactions: list[dict] = []

    if account_type == "checking":
        # Paychecks every other Friday-ish: simplified to day 1 and 15.
        for d in days:
            if d.day in (1, 15):
                merchant, description, amount = PAYCHECK
                transactions.append({
                    "merchant": merchant, "description": description,
                    "amount": round(amount + rng.uniform(-20, 20), 2),
                    "posted_at": d.isoformat(),
                })
        for merchant, description, day_of_month, amount in RECURRING_MONTHLY:
            for d in days:
                if d.day == day_of_month:
                    jitter = round(rng.uniform(-2.0, 2.0), 2) if amount > 20 else 0.0
                    transactions.append({
                        "merchant": merchant, "description": description,
                        "amount": -round(amount + jitter, 2),
                        "posted_at": d.isoformat(),
                    })

    # Discretionary spending: a handful of random purchases per week,
    # spread across all account types (checking + credit card both see
    # everyday purchases in a real setup; savings accounts do not).
    if account_type in ("checking", "credit_card"):
        for d in days:
            if rng.random() < 0.35:  # ~2-3 transactions/week
                merchant, description, (lo, hi) = rng.choice(DISCRETIONARY_SPENDS)
                amount = -round(rng.uniform(lo, hi), 2)
                transactions.append({
                    "merchant": merchant, "description": description,
                    "amount": amount, "posted_at": d.isoformat(),
                })

    if account_type == "savings":
        # Interest payment once a month, small transfer-in occasionally.
        for d in days:
            if d.day == 28:
                transactions.append({
                    "merchant": "Demo Bank", "description": "interest payment",
                    "amount": round(rng.uniform(0.50, 4.00), 2),
                    "posted_at": d.isoformat(),
                })
            elif rng.random() < 0.05:
                transactions.append({
                    "merchant": "Everyday Checking", "description": "transfer from checking",
                    "amount": round(rng.uniform(50, 400), 2),
                    "posted_at": d.isoformat(),
                })

    transactions.sort(key=lambda t: t["posted_at"])
    return transactions
