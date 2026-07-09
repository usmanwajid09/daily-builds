import pytest

from app import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    yield c
    c.close()


def test_create_and_get_workspace(conn):
    wid = db.create_workspace(conn, "Acme", "acme")
    conn.commit()
    w = db.get_workspace(conn, wid)
    assert w["name"] == "Acme"
    assert w["slug"] == "acme"


def test_workspace_slug_unique(conn):
    db.create_workspace(conn, "Acme", "acme")
    conn.commit()
    with pytest.raises(Exception):
        db.create_workspace(conn, "Acme Two", "acme")
        conn.commit()


def test_membership_unique_per_user_workspace(conn):
    wid = db.create_workspace(conn, "Acme", "acme")
    uid = db.create_user(conn, "a@example.com", "hash")
    conn.commit()
    db.create_membership(conn, uid, wid, "owner")
    conn.commit()
    with pytest.raises(Exception):
        db.create_membership(conn, uid, wid, "member")
        conn.commit()


def test_membership_role_check_constraint(conn):
    wid = db.create_workspace(conn, "Acme", "acme")
    uid = db.create_user(conn, "a@example.com", "hash")
    conn.commit()
    with pytest.raises(Exception):
        db.create_membership(conn, uid, wid, "superadmin")
        conn.commit()


def test_list_memberships_for_user_joins_workspace_name(conn):
    wid = db.create_workspace(conn, "Acme", "acme")
    uid = db.create_user(conn, "a@example.com", "hash")
    conn.commit()
    db.create_membership(conn, uid, wid, "owner")
    conn.commit()
    rows = db.list_memberships_for_user(conn, uid)
    assert len(rows) == 1
    assert rows[0]["workspace_slug"] == "acme"
    assert rows[0]["role"] == "owner"


def test_project_and_task_lifecycle(conn):
    wid = db.create_workspace(conn, "Acme", "acme")
    uid = db.create_user(conn, "a@example.com", "hash")
    conn.commit()
    pid = db.create_project(conn, wid, "Website Revamp")
    conn.commit()

    t1 = db.create_task(conn, pid, "Design mockups", "", "todo", uid)
    t2 = db.create_task(conn, pid, "Set up repo", "", "todo", uid)
    conn.commit()

    tasks = db.list_tasks_for_project(conn, pid)
    assert [t["id"] for t in tasks] == [t1, t2]
    assert tasks[0]["position"] == 0
    assert tasks[1]["position"] == 1  # next_position increments within (project, status)


def test_next_position_is_per_status_not_global(conn):
    wid = db.create_workspace(conn, "Acme", "acme")
    uid = db.create_user(conn, "a@example.com", "hash")
    conn.commit()
    pid = db.create_project(conn, wid, "P")
    conn.commit()

    db.create_task(conn, pid, "A", "", "todo", uid)
    db.create_task(conn, pid, "B", "", "todo", uid)
    conn.commit()
    # A fresh 'done' column should start at position 0, not continue from todo's count
    assert db.next_position(conn, pid, "done") == 0
    assert db.next_position(conn, pid, "todo") == 2


def test_update_task_partial_fields_only(conn):
    wid = db.create_workspace(conn, "Acme", "acme")
    uid = db.create_user(conn, "a@example.com", "hash")
    conn.commit()
    pid = db.create_project(conn, wid, "P")
    conn.commit()
    tid = db.create_task(conn, pid, "Title", "Desc", "todo", uid)
    conn.commit()

    db.update_task(conn, tid, status="in_progress")
    conn.commit()
    t = db.get_task(conn, tid)
    assert t["status"] == "in_progress"
    assert t["title"] == "Title"  # untouched
    assert t["description"] == "Desc"  # untouched


def test_update_task_with_no_fields_is_a_noop(conn):
    wid = db.create_workspace(conn, "Acme", "acme")
    uid = db.create_user(conn, "a@example.com", "hash")
    conn.commit()
    pid = db.create_project(conn, wid, "P")
    conn.commit()
    tid = db.create_task(conn, pid, "Title", "Desc", "todo", uid)
    conn.commit()
    db.update_task(conn, tid)  # should not raise / not touch SQL
    t = db.get_task(conn, tid)
    assert t["title"] == "Title"


def test_delete_task(conn):
    wid = db.create_workspace(conn, "Acme", "acme")
    uid = db.create_user(conn, "a@example.com", "hash")
    conn.commit()
    pid = db.create_project(conn, wid, "P")
    conn.commit()
    tid = db.create_task(conn, pid, "Title", "", "todo", uid)
    conn.commit()
    db.delete_task(conn, tid)
    conn.commit()
    assert db.get_task(conn, tid) is None


def test_transaction_rolls_back_on_exception(conn):
    db.create_workspace(conn, "Acme", "acme")
    conn.commit()
    with pytest.raises(ValueError):
        with db.transaction(conn):
            db.create_user(conn, "a@example.com", "hash")
            raise ValueError("boom")
    assert db.get_user_by_email(conn, "a@example.com") is None
