from datetime import date

from fintech_app import mock_data


def test_generate_mock_accounts_is_fixed_set():
    accounts = mock_data.generate_mock_accounts()
    types = {a["account_type"] for a in accounts}
    assert types == {"checking", "savings", "credit_card"}


def test_generate_mock_transactions_is_deterministic():
    today = date(2026, 7, 14)
    a = mock_data.generate_mock_transactions("checking", seed=42, months=3, today=today)
    b = mock_data.generate_mock_transactions("checking", seed=42, months=3, today=today)
    assert a == b


def test_different_seeds_produce_different_transactions():
    today = date(2026, 7, 14)
    a = mock_data.generate_mock_transactions("checking", seed=1, months=3, today=today)
    b = mock_data.generate_mock_transactions("checking", seed=2, months=3, today=today)
    assert a != b


def test_checking_account_has_recurring_rent_and_paycheck():
    today = date(2026, 7, 14)
    txns = mock_data.generate_mock_transactions("checking", seed=5, months=3, today=today)
    merchants = {t["merchant"] for t in txns}
    assert "Skyline Property Management" in merchants
    assert "Acme Corp Payroll" in merchants


def test_recurring_rent_appears_once_per_month():
    today = date(2026, 7, 14)
    txns = mock_data.generate_mock_transactions("checking", seed=5, months=3, today=today)
    rent_txns = [t for t in txns if t["merchant"] == "Skyline Property Management"]
    # 3 months back through today -> 3 or 4 rent payments depending on
    # where "today" falls relative to day 1 of the current month.
    assert 3 <= len(rent_txns) <= 4
    for t in rent_txns:
        assert t["posted_at"][8:10] == "01"


def test_savings_account_has_no_paycheck_or_rent():
    today = date(2026, 7, 14)
    txns = mock_data.generate_mock_transactions("savings", seed=5, months=3, today=today)
    merchants = {t["merchant"] for t in txns}
    assert "Acme Corp Payroll" not in merchants
    assert "Skyline Property Management" not in merchants


def test_transactions_sorted_by_posted_at():
    today = date(2026, 7, 14)
    txns = mock_data.generate_mock_transactions("credit_card", seed=7, months=2, today=today)
    dates = [t["posted_at"] for t in txns]
    assert dates == sorted(dates)


def test_all_transactions_have_required_keys():
    today = date(2026, 7, 14)
    txns = mock_data.generate_mock_transactions("checking", seed=9, months=1, today=today)
    for t in txns:
        assert set(t.keys()) == {"merchant", "description", "amount", "posted_at"}
        assert isinstance(t["amount"], float)
