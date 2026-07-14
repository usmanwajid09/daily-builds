from flask import Blueprint, current_app, g, jsonify, request

from .. import categorization, db
from ..decorators import require_auth

bp = Blueprint("account_routes", __name__, url_prefix="/api/accounts")

VALID_ACCOUNT_TYPES = {"checking", "savings", "credit_card"}


def _account_owned_by(conn, account_id: int, user_id: int):
    """Return the account row if it exists and belongs to user_id, else
    None. Callers return 404 (not 403) on a miss so a user can't probe
    which account ids exist for someone else."""
    account = db.get_account(conn, account_id)
    if account is None or account["user_id"] != user_id:
        return None
    return account


def _account_to_dict(conn, account) -> dict:
    d = db.row_to_dict(account)
    d["balance"] = db.account_balance(conn, account["id"])
    return d


@bp.get("")
@require_auth
def list_accounts():
    conn = current_app.config["DB_CONN"]
    accounts = db.list_accounts_for_user(conn, g.user_id)
    return jsonify(accounts=[_account_to_dict(conn, a) for a in accounts])


@bp.post("")
@require_auth
def create_account():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    account_type = (body.get("account_type") or "").strip()
    institution_name = (body.get("institution_name") or "Demo Bank").strip()
    currency = (body.get("currency") or "USD").strip().upper()

    if not name:
        return jsonify(error="name is required"), 400
    if account_type not in VALID_ACCOUNT_TYPES:
        return jsonify(error=f"account_type must be one of {sorted(VALID_ACCOUNT_TYPES)}"), 400

    conn = current_app.config["DB_CONN"]
    with db.transaction(conn):
        account_id = db.create_account(
            conn, g.user_id, name, account_type, institution_name, currency, is_mock=False,
        )
    return jsonify(_account_to_dict(conn, db.get_account(conn, account_id))), 201


@bp.get("/<int:account_id>")
@require_auth
def get_account(account_id):
    conn = current_app.config["DB_CONN"]
    account = _account_owned_by(conn, account_id, g.user_id)
    if account is None:
        return jsonify(error="account not found"), 404
    return jsonify(_account_to_dict(conn, account))


@bp.delete("/<int:account_id>")
@require_auth
def delete_account(account_id):
    """Owner-only. Cascades to the account's transactions via ON DELETE
    CASCADE."""
    conn = current_app.config["DB_CONN"]
    account = _account_owned_by(conn, account_id, g.user_id)
    if account is None:
        return jsonify(error="account not found"), 404

    with db.transaction(conn):
        db.delete_account(conn, account_id)
    return jsonify(deleted=True, account_id=account_id)


@bp.get("/<int:account_id>/transactions")
@require_auth
def list_account_transactions(account_id):
    conn = current_app.config["DB_CONN"]
    account = _account_owned_by(conn, account_id, g.user_id)
    if account is None:
        return jsonify(error="account not found"), 404

    category = request.args.get("category")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    rows = db.list_transactions_for_account(conn, account_id, category, start_date, end_date)
    return jsonify(transactions=[db.row_to_dict(r) for r in rows])


@bp.post("/<int:account_id>/transactions")
@require_auth
def create_account_transaction(account_id):
    conn = current_app.config["DB_CONN"]
    account = _account_owned_by(conn, account_id, g.user_id)
    if account is None:
        return jsonify(error="account not found"), 404

    body = request.get_json(silent=True) or {}
    merchant = (body.get("merchant") or "").strip()
    description = (body.get("description") or "").strip()
    posted_at = (body.get("posted_at") or "").strip()
    category_override = (body.get("category") or "").strip() or None

    try:
        amount = float(body.get("amount"))
    except (TypeError, ValueError):
        return jsonify(error="amount must be a number"), 400
    if not posted_at:
        return jsonify(error="posted_at (ISO date) is required"), 400
    if category_override and category_override not in categorization.ALL_CATEGORIES:
        return jsonify(error=f"category must be one of {categorization.ALL_CATEGORIES}"), 400

    if category_override:
        category = category_override
        is_manual = True
    else:
        category = categorization.categorize(merchant, description, amount)
        is_manual = False

    with db.transaction(conn):
        transaction_id = db.create_transaction(
            conn, account_id, amount, merchant, description, category, posted_at, is_manual,
        )
    return jsonify(db.row_to_dict(db.get_transaction(conn, transaction_id))), 201
