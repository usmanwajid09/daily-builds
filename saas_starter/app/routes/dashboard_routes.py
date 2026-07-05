"""Tenant-scoped dashboard skeleton.

Every query here is filtered by g.tenant_id, which comes only from the
verified JWT -- never from a client-supplied id -- so there's no request
parameter an attacker could tamper with to see another tenant's data.
"""
from flask import Blueprint, current_app, g, jsonify

from .. import db
from ..decorators import jwt_required

bp = Blueprint("dashboard", __name__, url_prefix="/api")


@bp.get("/dashboard")
@jwt_required
def dashboard():
    db_path = current_app.config["DB_PATH"]
    with db.connect(db_path) as conn:
        tenant = db.get_tenant(conn, g.tenant_id)
        members = db.list_members_for_tenant(conn, g.tenant_id)

    # Stub metrics -- real usage/billing metrics are out of scope for this
    # milestone (billing itself is a later milestone, Stripe TEST mode only).
    return jsonify(
        tenant={"id": tenant["id"], "name": tenant["name"], "slug": tenant["slug"]},
        plan="free",
        member_count=len(members),
        your_role=g.role,
        widgets=[
            {"label": "Team members", "value": len(members)},
            {"label": "Plan", "value": "free"},
            {"label": "Projects", "value": 0},
        ],
    )


@bp.get("/tenants/me/members")
@jwt_required
def list_members():
    db_path = current_app.config["DB_PATH"]
    with db.connect(db_path) as conn:
        members = db.list_members_for_tenant(conn, g.tenant_id)
    return jsonify(members=[
        {"id": m["id"], "email": m["email"], "role": m["role"], "joined_at": m["created_at"]}
        for m in members
    ])
