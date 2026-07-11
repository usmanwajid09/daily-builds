from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..decorators import require_auth
from ..mentions import extract_mentioned_emails

bp = Blueprint("comment_routes", __name__, url_prefix="/api")


def _task_in_scope(conn, task_id):
    """Returns the task row, or None if it doesn't exist or its project
    belongs to a different workspace than the caller's token."""
    task = db.get_task(conn, task_id)
    if task is None:
        return None
    project = db.get_project(conn, task["project_id"])
    if project is None or project["workspace_id"] != g.workspace_id:
        return None
    return task


def _broadcaster():
    return current_app.config["BROADCASTER"]


def _notify_mentions(conn, body: str, task_id: int, comment_id: int):
    """Creates a notification (+ a real-time push if that user has a
    live WS connection subscribed to their notification channel) for
    every @email mention in `body` that resolves to an actual workspace
    member other than the comment's own author. Mentions of emails that
    aren't workspace members are silently ignored."""
    mentioned_emails = extract_mentioned_emails(body)
    if not mentioned_emails:
        return

    members_by_email = {m["email"]: m for m in db.list_members_for_workspace(conn, g.workspace_id)}
    author = db.get_user(conn, g.user_id)

    for email in mentioned_emails:
        member = members_by_email.get(email)
        if member is None or member["id"] == g.user_id:
            continue  # not a member, or mentioning yourself -- no notification either way

        message = f"{author['email']} mentioned you in a comment"
        with db.transaction(conn):
            notif_id = db.create_notification(
                conn, user_id=member["id"], workspace_id=g.workspace_id,
                type_="mention", message=message, actor_id=g.user_id,
                task_id=task_id, comment_id=comment_id,
            )
        notif = db.row_to_dict(db.get_notification(conn, notif_id))
        _broadcaster().broadcast(f"user:{member['id']}", {"type": "notification", "notification": notif})


@bp.get("/tasks/<int:task_id>/comments")
@require_auth
def list_comments(task_id):
    conn = current_app.config["DB_CONN"]
    if _task_in_scope(conn, task_id) is None:
        return jsonify(error="task not found"), 404
    comments = db.list_comments_for_task(conn, task_id)
    return jsonify(comments=[db.row_to_dict(c) for c in comments])


@bp.post("/tasks/<int:task_id>/comments")
@require_auth
def create_comment(task_id):
    conn = current_app.config["DB_CONN"]
    task = _task_in_scope(conn, task_id)
    if task is None:
        return jsonify(error="task not found"), 404

    body_json = request.get_json(silent=True) or {}
    body = (body_json.get("body") or "").strip()
    if not body:
        return jsonify(error="body is required"), 400

    with db.transaction(conn):
        comment_id = db.create_comment(conn, task_id, g.user_id, body)
    comment = db.row_to_dict(db.get_comment(conn, comment_id))

    _broadcaster().broadcast(task["project_id"], {
        "type": "comment_created", "project_id": task["project_id"],
        "task_id": task_id, "comment": comment,
    })

    _notify_mentions(conn, body, task_id, comment_id)

    return jsonify(comment), 201


@bp.delete("/comments/<int:comment_id>")
@require_auth
def delete_comment(comment_id):
    conn = current_app.config["DB_CONN"]
    comment = db.get_comment(conn, comment_id)
    if comment is None:
        return jsonify(error="comment not found"), 404
    task = _task_in_scope(conn, comment["task_id"])
    if task is None:
        return jsonify(error="comment not found"), 404

    # Authors can delete their own comments; owner/admin can moderate anyone's.
    if comment["author_id"] != g.user_id and g.role not in ("owner", "admin"):
        return jsonify(error="not allowed to delete this comment"), 403

    with db.transaction(conn):
        db.delete_comment(conn, comment_id)

    _broadcaster().broadcast(task["project_id"], {
        "type": "comment_deleted", "project_id": task["project_id"],
        "task_id": task["id"], "comment_id": comment_id,
    })
    return jsonify(deleted=True, comment_id=comment_id)
