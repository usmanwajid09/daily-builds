"""SQLite data-access layer for the multi-tenant SaaS starter.

Data model
----------
tenants      -- one row per organization/workspace.
users        -- global user identities (an email is unique across the
                whole system, not per-tenant). A user's password lives
                here since credentials are tenant-independent.
memberships  -- join table between users and tenants. A single user can
                belong to more than one tenant, each with its own role
                (e.g. "owner", "member"). This is the piece that makes
                the model genuinely multi-tenant rather than just
                "one account per company".

All tenant-scoped queries take a tenant_id and filter on it explicitly,
so cross-tenant data leaks are structural mistakes, not accidental ones.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone


SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memberships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    tenant_id INTEGER NOT NULL REFERENCES tenants(id),
    role TEXT NOT NULL CHECK(role IN ('owner', 'admin', 'member')),
    created_at TEXT NOT NULL,
    UNIQUE(user_id, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_memberships_tenant ON memberships(tenant_id);
CREATE INDEX IF NOT EXISTS idx_memberships_user ON memberships(user_id);

CREATE TABLE IF NOT EXISTS subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id INTEGER NOT NULL UNIQUE REFERENCES tenants(id),
    plan TEXT NOT NULL DEFAULT 'free' CHECK(plan IN ('free', 'pro')),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'pending', 'canceled')),
    stripe_customer_id TEXT,
    stripe_checkout_session_id TEXT,
    updated_at TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect(db_path: str):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --- Tenant queries -------------------------------------------------------

def create_tenant(conn: sqlite3.Connection, name: str, slug: str) -> int:
    cur = conn.execute(
        "INSERT INTO tenants (name, slug, created_at) VALUES (?, ?, ?)",
        (name, slug, now_iso()),
    )
    return cur.lastrowid


def get_tenant(conn: sqlite3.Connection, tenant_id: int):
    return conn.execute(
        "SELECT * FROM tenants WHERE id = ?", (tenant_id,)
    ).fetchone()


def get_tenant_by_slug(conn: sqlite3.Connection, slug: str):
    return conn.execute(
        "SELECT * FROM tenants WHERE slug = ?", (slug,)
    ).fetchone()


# --- User queries ----------------------------------------------------------

def create_user(conn: sqlite3.Connection, email: str, password_hash: str) -> int:
    cur = conn.execute(
        "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
        (email, password_hash, now_iso()),
    )
    return cur.lastrowid


def get_user_by_email(conn: sqlite3.Connection, email: str):
    return conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()


def get_user_by_id(conn: sqlite3.Connection, user_id: int):
    return conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()


# --- Membership queries ------------------------------------------------------

def create_membership(conn: sqlite3.Connection, user_id: int, tenant_id: int, role: str) -> int:
    cur = conn.execute(
        "INSERT INTO memberships (user_id, tenant_id, role, created_at) VALUES (?, ?, ?, ?)",
        (user_id, tenant_id, role, now_iso()),
    )
    return cur.lastrowid


def get_membership(conn: sqlite3.Connection, user_id: int, tenant_id: int):
    return conn.execute(
        "SELECT * FROM memberships WHERE user_id = ? AND tenant_id = ?",
        (user_id, tenant_id),
    ).fetchone()


def list_memberships_for_user(conn: sqlite3.Connection, user_id: int):
    return conn.execute(
        """SELECT m.*, t.name AS tenant_name, t.slug AS tenant_slug
           FROM memberships m JOIN tenants t ON t.id = m.tenant_id
           WHERE m.user_id = ?""",
        (user_id,),
    ).fetchall()


def list_members_for_tenant(conn: sqlite3.Connection, tenant_id: int):
    """Tenant-scoped by construction: always filtered on tenant_id."""
    return conn.execute(
        """SELECT u.id, u.email, m.role, m.created_at
           FROM memberships m JOIN users u ON u.id = m.user_id
           WHERE m.tenant_id = ?
           ORDER BY m.created_at ASC""",
        (tenant_id,),
    ).fetchall()


# --- Subscription queries (Milestone 2: billing) ---------------------------
#
# Every tenant gets exactly one subscriptions row, created at signup
# (defaulting to the 'free' plan) by create_default_subscription. Billing
# routes then update that same row rather than inserting new ones, so
# "get a tenant's plan" is always a single-row lookup, never an
# aggregation over history.

def create_default_subscription(conn: sqlite3.Connection, tenant_id: int) -> int:
    cur = conn.execute(
        "INSERT INTO subscriptions (tenant_id, plan, status, updated_at) "
        "VALUES (?, 'free', 'active', ?)",
        (tenant_id, now_iso()),
    )
    return cur.lastrowid


def get_subscription(conn: sqlite3.Connection, tenant_id: int):
    return conn.execute(
        "SELECT * FROM subscriptions WHERE tenant_id = ?", (tenant_id,)
    ).fetchone()


def set_subscription_pending(
    conn: sqlite3.Connection, tenant_id: int, plan: str, stripe_checkout_session_id: str
) -> None:
    """Mark a tenant's subscription as pending a specific plan upgrade while
    checkout is in progress (before the webhook confirms payment)."""
    conn.execute(
        "UPDATE subscriptions SET plan = ?, status = 'pending', "
        "stripe_checkout_session_id = ?, updated_at = ? WHERE tenant_id = ?",
        (plan, stripe_checkout_session_id, now_iso(), tenant_id),
    )


def activate_subscription(
    conn: sqlite3.Connection, tenant_id: int, plan: str, stripe_customer_id: str | None = None
) -> None:
    """Called from the billing webhook once checkout actually completes."""
    conn.execute(
        "UPDATE subscriptions SET plan = ?, status = 'active', "
        "stripe_customer_id = COALESCE(?, stripe_customer_id), updated_at = ? "
        "WHERE tenant_id = ?",
        (plan, stripe_customer_id, now_iso(), tenant_id),
    )


def get_subscription_by_checkout_session(conn: sqlite3.Connection, session_id: str):
    return conn.execute(
        "SELECT * FROM subscriptions WHERE stripe_checkout_session_id = ?", (session_id,)
    ).fetchone()
