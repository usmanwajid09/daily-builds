import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db  # noqa: E402


def _fresh_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


def test_init_db_creates_tables():
    path = _fresh_db_path()
    try:
        db.init_db(path)
        with db.connect(path) as conn:
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert {"tenants", "users", "memberships"} <= tables
    finally:
        os.remove(path)


def test_tenant_slug_unique():
    path = _fresh_db_path()
    try:
        db.init_db(path)
        with db.connect(path) as conn:
            db.create_tenant(conn, "Acme", "acme")
        with db.connect(path) as conn:
            try:
                db.create_tenant(conn, "Acme Two", "acme")
                assert False, "expected a uniqueness violation"
            except Exception:
                pass
    finally:
        os.remove(path)


def test_membership_join_returns_tenant_fields():
    path = _fresh_db_path()
    try:
        db.init_db(path)
        with db.connect(path) as conn:
            tenant_id = db.create_tenant(conn, "Acme", "acme")
            user_id = db.create_user(conn, "a@example.com", "hash")
            db.create_membership(conn, user_id, tenant_id, "owner")

        with db.connect(path) as conn:
            rows = db.list_memberships_for_user(conn, user_id)
        assert len(rows) == 1
        assert rows[0]["tenant_slug"] == "acme"
        assert rows[0]["role"] == "owner"
    finally:
        os.remove(path)


def test_list_members_for_tenant_is_scoped():
    path = _fresh_db_path()
    try:
        db.init_db(path)
        with db.connect(path) as conn:
            t1 = db.create_tenant(conn, "Acme", "acme")
            t2 = db.create_tenant(conn, "Globex", "globex")
            u1 = db.create_user(conn, "a@example.com", "hash")
            u2 = db.create_user(conn, "b@example.com", "hash")
            db.create_membership(conn, u1, t1, "owner")
            db.create_membership(conn, u2, t2, "owner")

        with db.connect(path) as conn:
            members_t1 = db.list_members_for_tenant(conn, t1)
            members_t2 = db.list_members_for_tenant(conn, t2)

        assert [m["email"] for m in members_t1] == ["a@example.com"]
        assert [m["email"] for m in members_t2] == ["b@example.com"]
    finally:
        os.remove(path)
