from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..decorators import require_auth

bp = Blueprint("notification_routes", __name__, url_prefix="/api/notifications")


@bp.get("")
@require_auth
def list_notifications():
    conn = current_app.config["DB_CONN"]
    unread_only = request.args.get("unread") == "1"
    notifications = db.list_notifications_for_user(conn, g.user_id, g.workspace_id, unread_only=unread_only)
    return jsonify(
        notifications=[db.row_to_dict(n) for n in notifications],
        unread_count=db.count_unread_notifications(conn, g.user_id, g.workspace_id),
    )


@bp.post("/<int:notification_id>/read")
@require_auth
def mark_read(notification_id):
    conn = current_app.config["DB_CONN"]
    notif = db.get_notification(conn, notification_id)
    if notif is None or notif["user_id"] != g.user_id:
        return jsonify(error="notification not found"), 404
    with db.transaction(conn):
        db.mark_notification_read(conn, notification_id)
    return jsonify(db.row_to_dict(db.get_notification(conn, notification_id)))


@bp.post("/read-all")
@require_auth
def mark_all_read():
    conn = current_app.config["DB_CONN"]
    with db.transaction(conn):
        count = db.mark_all_notifications_read(conn, g.user_id, g.workspace_id)
    return jsonify(marked=count)
