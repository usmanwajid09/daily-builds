"""Signup / login / session-introspection / tenant-invite endpoints."""
import secrets
import sqlite3

from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..auth import (
    hash_password,
    is_valid_email,
    issue_token,
    slugify,
    verify_password,
)
from ..decorators import jwt_required, role_required

bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@bp.post("/signup")
def signup():
    """Create a brand-new tenant plus its first user (role=owner).

    This is the "sign up = create your organization" flow common to SaaS
    starters. Joining an *existing* tenant happens via /invite instead.
    """
    body = request.get_json(silent=True) or {}
    org_name = (body.get("org_name") or "").strip()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not org_name:
        return jsonify(error="org_name is required"), 400
    if not is_valid_email(email):
        return jsonify(error="a valid email is required"), 400
    try:
        password_hash = hash_password(password)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    db_path = current_app.config["DB_PATH"]
    try:
        with db.connect(db_path) as conn:
            if db.get_user_by_email(conn, email) is not None:
                return jsonify(error="email already registered"), 409

            slug = slugify(org_name)
            base_slug = slug
            suffix = 1
            while db.get_tenant_by_slug(conn, slug) is not None:
                suffix += 1
                slug = f"{base_slug}-{suffix}"

            tenant_id = db.create_tenant(conn, org_name, slug)
            user_id = db.create_user(conn, email, password_hash)
            db.create_membership(conn, user_id, tenant_id, role="owner")
    except sqlite3.IntegrityError:
        # A concurrent signup won the race on the same email or slug between
        # our check above and this transaction's commit. Rare, but with two
        # requests racing on check-then-insert it's a real possibility, not
        # hypothetical -- surface it as a normal conflict rather than a 500.
        return jsonify(error="email or organization slug already taken, please retry"), 409

    token = issue_token(current_app.config["JWT_SECRET"], user_id, tenant_id, "owner")
    return jsonify(
        token=token,
        user={"id": user_id, "email": email},
        tenant={"id": tenant_id, "name": org_name, "slug": slug},
        role="owner",
    ), 201


@bp.post("/login")
def login():
    """Log in with email + password. If the account belongs to more than
    one tenant, the caller must supply `tenant_slug` to disambiguate which
    org context the issued token should be scoped to.
    """
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    tenant_slug = (body.get("tenant_slug") or "").strip() or None

    db_path = current_app.config["DB_PATH"]
    with db.connect(db_path) as conn:
        user = db.get_user_by_email(conn, email)
        if user is None or not verify_password(password, user["password_hash"]):
            return jsonify(error="invalid email or password"), 401

        memberships = db.list_memberships_for_user(conn, user["id"])
        if not memberships:
            return jsonify(error="account has no tenant memberships"), 403

        if tenant_slug:
            match = next((m for m in memberships if m["tenant_slug"] == tenant_slug), None)
            if match is None:
                return jsonify(error="no membership in that tenant"), 403
            membership = match
        elif len(memberships) == 1:
            membership = memberships[0]
        else:
            return jsonify(
                error="account belongs to multiple tenants; specify tenant_slug",
                tenants=[{"slug": m["tenant_slug"], "name": m["tenant_name"]} for m in memberships],
            ), 409

    token = issue_token(
        current_app.config["JWT_SECRET"], user["id"], membership["tenant_id"], membership["role"]
    )
    return jsonify(
        token=token,
        user={"id": user["id"], "email": email},
        tenant={"id": membership["tenant_id"], "slug": membership["tenant_slug"], "name": membership["tenant_name"]},
        role=membership["role"],
    ), 200


@bp.get("/me")
@jwt_required
def me():
    db_path = current_app.config["DB_PATH"]
    with db.connect(db_path) as conn:
        user = db.get_user_by_id(conn, g.current_user_id)
        tenant = db.get_tenant(conn, g.tenant_id)
    return jsonify(
        user={"id": user["id"], "email": user["email"]},
        tenant={"id": tenant["id"], "name": tenant["name"], "slug": tenant["slug"]},
        role=g.role,
    )


@bp.post("/invite")
@jwt_required
@role_required("owner", "admin")
def invite():
    """Invite a user (by email) into the caller's current tenant.

    If the email has no account yet, one is created with a randomly
    generated temporary password that is returned in the response body
    -- in a real product this would be emailed via a signed invite link
    instead of handed back in the API response. Dev/demo simplification,
    called out in the README.
    """
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    role = (body.get("role") or "member").strip()

    if not is_valid_email(email):
        return jsonify(error="a valid email is required"), 400
    if role not in ("admin", "member"):
        return jsonify(error="role must be 'admin' or 'member'"), 400

    db_path = current_app.config["DB_PATH"]
    temp_password = None
    try:
        with db.connect(db_path) as conn:
            user = db.get_user_by_email(conn, email)
            if user is None:
                temp_password = secrets.token_urlsafe(9)
                user_id = db.create_user(conn, email, hash_password(temp_password))
            else:
                user_id = user["id"]
                if db.get_membership(conn, user_id, g.tenant_id) is not None:
                    return jsonify(error="user is already a member of this tenant"), 409

            db.create_membership(conn, user_id, g.tenant_id, role=role)
    except sqlite3.IntegrityError:
        # Same race-condition safety net as signup: two concurrent invites
        # for the same email+tenant would otherwise surface as a 500.
        return jsonify(error="user is already a member of this tenant"), 409

    resp = {"user_id": user_id, "email": email, "role": role}
    if temp_password:
        resp["temp_password"] = temp_password
        resp["note"] = "dev-only: in production this would be an emailed invite link, not a returned password"
    return jsonify(**resp), 201
