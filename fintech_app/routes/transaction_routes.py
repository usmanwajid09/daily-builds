from flask import Blueprint, current_app, g, jsonify, request

from .. import categorization, db
from ..decorators import require_auth

bp = Blueprint("transaction_routes", __name__, url_prefix="/api/transactions")


def _transaction_owned_by(conn, transaction_id: int, user_id: int):
    """Return the transaction row if it exists and its account belongs to
    user_id, else None (404-not-403, same pattern as accounts)."""
    txn = db.get_transaction(conn, transaction_id)
    if txn is None:
        return None
    account = db.get_account(conn, txn["account_id"])
    if account is None or account["user_id"] != user_id:
        return None
    return txn


@bp.get("")
@require_auth
def list_transactions():
    """All transactions across every account owned by the caller."""
    conn = current_app.config["DB_CONN"]
    category = request.args.get("category")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    rows = db.list_transactions_for_user(conn, g.user_id, category, start_date, end_date)
    return jsonify(transactions=[db.row_to_dict(r) for r in rows])


@bp.get("/categories")
@require_auth
def list_categories():
    return jsonify(categories=categorization.ALL_CATEGORIES)


@bp.get("/<int:transaction_id>")
@require_auth
def get_transaction(transaction_id):
    conn = current_app.config["DB_CONN"]
    txn = _transaction_owned_by(conn, transaction_id, g.user_id)
    if txn is None:
        return jsonify(error="transaction not found"), 404
    return jsonify(db.row_to_dict(txn))


@bp.patch("/<int:transaction_id>")
@require_auth
def update_transaction(transaction_id):
    """Currently supports recategorizing a transaction. A manual
    recategorization is sticky: future re-runs of the auto-categorizer
    (POST /api/demo/recategorize) skip any transaction with
    category_is_manual set."""
    conn = current_app.config["DB_CONN"]
    txn = _transaction_owned_by(conn, transaction_id, g.user_id)
    if txn is None:
        return jsonify(error="transaction not found"), 404

    body = request.get_json(silent=True) or {}
    if "category" not in body:
        return jsonify(error="category is required"), 400
    category = (body.get("category") or "").strip().lower()
    if category not in categorization.ALL_CATEGORIES:
        return jsonify(error=f"category must be one of {categorization.ALL_CATEGORIES}"), 400

    with db.transaction(conn):
        db.update_transaction_category(conn, transaction_id, category, category_is_manual=True)
    return jsonify(db.row_to_dict(db.get_transaction(conn, transaction_id)))


@bp.delete("/<int:transaction_id>")
@require_auth
def delete_transaction(transaction_id):
    conn = current_app.config["DB_CONN"]
    txn = _transaction_owned_by(conn, transaction_id, g.user_id)
    if txn is None:
        return jsonify(error="transaction not found"), 404

    with db.transaction(conn):
        db.delete_transaction(conn, transaction_id)
    return jsonify(deleted=True, transaction_id=transaction_id)
