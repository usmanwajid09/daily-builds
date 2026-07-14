from flask import Blueprint, current_app, g, jsonify, request

from .. import categorization, db, mock_data
from ..decorators import require_auth

bp = Blueprint("demo_routes", __name__, url_prefix="/api/demo")


@bp.post("/seed")
@require_auth
def seed_demo_data():
    """Create 3 mock accounts with a few months of seeded synthetic
    transactions for the caller, each transaction run through the same
    categorization engine a real manually-entered transaction would use.

    NEVER touches a real bank -- this is the only "data ingestion" path
    in the app, and it is entirely synthetic. See mock_data.py.

    Safe to call more than once: pass {"reset": true} to first delete the
    caller's existing mock accounts (and their transactions, via cascade)
    before reseeding, otherwise repeated calls just add more mock
    accounts alongside any manually-created ones.
    """
    conn = current_app.config["DB_CONN"]
    body = request.get_json(silent=True) or {}
    reset = bool(body.get("reset", False))
    months = body.get("months", 3)
    try:
        months = int(months)
    except (TypeError, ValueError):
        return jsonify(error="months must be an integer"), 400
    if not (1 <= months <= 12):
        return jsonify(error="months must be between 1 and 12"), 400

    with db.transaction(conn):
        if reset:
            for acc in db.list_accounts_for_user(conn, g.user_id):
                if acc["is_mock"]:
                    db.delete_account(conn, acc["id"])

        created_accounts = []
        for i, acc_spec in enumerate(mock_data.generate_mock_accounts()):
            account_id = db.create_account(
                conn, g.user_id, acc_spec["name"], acc_spec["account_type"],
                acc_spec["institution_name"], "USD", is_mock=True,
            )
            txns = mock_data.generate_mock_transactions(
                acc_spec["account_type"], seed=g.user_id * 100 + i, months=months,
            )
            for t in txns:
                category = categorization.categorize(t["merchant"], t["description"], t["amount"])
                db.create_transaction(
                    conn, account_id, t["amount"], t["merchant"], t["description"],
                    category, t["posted_at"], category_is_manual=False,
                )
            created_accounts.append(account_id)

    accounts = [db.row_to_dict(db.get_account(conn, aid)) for aid in created_accounts]
    total_txns = sum(
        len(db.list_transactions_for_account(conn, aid)) for aid in created_accounts
    )
    return jsonify(accounts=accounts, transactions_created=total_txns), 201


@bp.post("/recategorize")
@require_auth
def recategorize():
    """Re-run the categorization engine over every transaction owned by
    the caller that was never manually recategorized. Useful after the
    keyword rules change, without disturbing anything a user explicitly
    fixed by hand."""
    conn = current_app.config["DB_CONN"]
    rows = db.list_uncategorized_or_auto_transactions(conn, g.user_id)
    updated = 0
    with db.transaction(conn):
        for row in rows:
            new_category = categorization.categorize(row["merchant"], row["description"], row["amount"])
            if new_category != row["category"]:
                db.update_transaction_category(conn, row["id"], new_category, category_is_manual=False)
                updated += 1
    return jsonify(checked=len(rows), updated=updated)
