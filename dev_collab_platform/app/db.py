"""SQLite schema + connection helpers for the dev-collab-platform.

Data model
----------
workspaces      -- one row per organization/team
users           -- global user identity (an email can belong to many workspaces)
memberships     -- join table: user_id + workspace_id -> role
projects        -- belong to exactly one workspace
tasks           -- belong to exactly one project; status is the task-board column
comments        -- belong to exactly one task; @email mentions notify workspace members
notifications   -- per-user, per-workspace; created by mentions and role changes so far

Every workspace-scoped query filters by workspace_id, and that
workspace_id always comes from the verified JWT on the request/WS
connection -- never from a client-supplied field -- so there is no
request parameter that lets one workspace read or write another's data.

Deleting a project cascades to its tasks, which cascades to their
comments and any notifications that reference those tasks/comments
(ON DELETE CASCADE below) -- deleting a project cleans up everything
under it in one statement rather than needing application-level cleanup.
"""
import sqlite3
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memberships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
    role TEXT NOT NULL CHECK(role IN ('owner', 'admin', 'member')),
    created_at REAL NOT NULL,
    UNIQUE(user_id, workspace_id)
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'todo' CHECK(status IN ('todo', 'in_progress', 'done')),
    position INTEGER NOT NULL DEFAULT 0,
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    author_id INTEGER NOT NULL REFERENCES users(id),
    body TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
    type TEXT NOT NULL CHECK(type IN ('mention', 'role_changed')),
    message TEXT NOT NULL,
    task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
    comment_id INTEGER REFERENCES comments(id) ON DELETE CASCADE,
    actor_id INTEGER REFERENCES users(id),
    read_at REAL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memberships_user ON memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_memberships_workspace ON memberships(workspace_id);
CREATE INDEX IF NOT EXISTS idx_projects_workspace ON projects(workspace_id);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_comments_task ON comments(task_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id, workspace_id);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """One connection per caller/thread is the intended usage pattern --
    see app/ws/server.py's conn_factory. For file-backed databases,
    SQLite's own file locking (with WAL + a busy_timeout) safely
    arbitrates between independent connections from different threads
    pointed at the same path; check_same_thread=False just tells the
    Python wrapper not to second-guess that. For ':memory:' paths (unit
    tests) there is only ever one connection anyway, so the pragmas are
    no-ops."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass  # :memory: databases don't support WAL; default mode is fine
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def now() -> float:
    return time.time()


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Commit on success, rollback on any exception."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------- workspaces
def create_workspace(conn, name: str, slug: str):
    cur = conn.execute(
        "INSERT INTO workspaces (name, slug, created_at) VALUES (?, ?, ?)",
        (name, slug, now()),
    )
    return cur.lastrowid


def get_workspace(conn, workspace_id: int):
    return conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()


def get_workspace_by_slug(conn, slug: str):
    return conn.execute("SELECT * FROM workspaces WHERE slug = ?", (slug,)).fetchone()


# --------------------------------------------------------------------- users
def create_user(conn, email: str, password_hash: str):
    cur = conn.execute(
        "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
        (email, password_hash, now()),
    )
    return cur.lastrowid


def get_user_by_email(conn, email: str):
    return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


def get_user(conn, user_id: int):
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


# --------------------------------------------------------------- memberships
def create_membership(conn, user_id: int, workspace_id: int, role: str):
    cur = conn.execute(
        "INSERT INTO memberships (user_id, workspace_id, role, created_at) VALUES (?, ?, ?, ?)",
        (user_id, workspace_id, role, now()),
    )
    return cur.lastrowid


def get_membership(conn, user_id: int, workspace_id: int):
    return conn.execute(
        "SELECT * FROM memberships WHERE user_id = ? AND workspace_id = ?",
        (user_id, workspace_id),
    ).fetchone()


def list_memberships_for_user(conn, user_id: int):
    return conn.execute(
        """SELECT m.*, w.name AS workspace_name, w.slug AS workspace_slug
           FROM memberships m JOIN workspaces w ON w.id = m.workspace_id
           WHERE m.user_id = ?""",
        (user_id,),
    ).fetchall()


def list_members_for_workspace(conn, workspace_id: int):
    return conn.execute(
        """SELECT u.id, u.email, m.role, m.created_at
           FROM memberships m JOIN users u ON u.id = m.user_id
           WHERE m.workspace_id = ?
           ORDER BY m.created_at ASC""",
        (workspace_id,),
    ).fetchall()


def update_membership_role(conn, user_id: int, workspace_id: int, role: str):
    conn.execute(
        "UPDATE memberships SET role = ? WHERE user_id = ? AND workspace_id = ?",
        (role, user_id, workspace_id),
    )


def delete_membership(conn, user_id: int, workspace_id: int):
    conn.execute(
        "DELETE FROM memberships WHERE user_id = ? AND workspace_id = ?",
        (user_id, workspace_id),
    )


def count_owners(conn, workspace_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM memberships WHERE workspace_id = ? AND role = 'owner'",
        (workspace_id,),
    ).fetchone()
    return row["n"]


# ------------------------------------------------------------------ projects
def create_project(conn, workspace_id: int, name: str):
    cur = conn.execute(
        "INSERT INTO projects (workspace_id, name, created_at) VALUES (?, ?, ?)",
        (workspace_id, name, now()),
    )
    return cur.lastrowid


def get_project(conn, project_id: int):
    return conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()


def list_projects_for_workspace(conn, workspace_id: int):
    return conn.execute(
        "SELECT * FROM projects WHERE workspace_id = ? ORDER BY created_at ASC",
        (workspace_id,),
    ).fetchall()


def delete_project(conn, project_id: int):
    """Cascades to tasks -> comments -> notifications via ON DELETE
    CASCADE foreign keys (see SCHEMA)."""
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))


# --------------------------------------------------------------------- tasks
def next_position(conn, project_id: int, status: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(position), -1) AS max_pos FROM tasks WHERE project_id = ? AND status = ?",
        (project_id, status),
    ).fetchone()
    return row["max_pos"] + 1


def create_task(conn, project_id: int, title: str, description: str, status: str,
                 created_by: int, position: int = None):
    ts = now()
    if position is None:
        position = next_position(conn, project_id, status)
    cur = conn.execute(
        """INSERT INTO tasks (project_id, title, description, status, position,
                               created_by, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (project_id, title, description, status, position, created_by, ts, ts),
    )
    return cur.lastrowid


def get_task(conn, task_id: int):
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


def list_tasks_for_project(conn, project_id: int):
    return conn.execute(
        "SELECT * FROM tasks WHERE project_id = ? ORDER BY status ASC, position ASC",
        (project_id,),
    ).fetchall()


def update_task(conn, task_id: int, **fields):
    """Update whichever of title/description/status/position are provided."""
    allowed = {"title", "description", "status", "position"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return
    updates["updated_at"] = now()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [task_id]
    conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", values)


def delete_task(conn, task_id: int):
    """Cascades to comments -> notifications via ON DELETE CASCADE."""
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))


# ------------------------------------------------------------------ comments
def create_comment(conn, task_id: int, author_id: int, body: str):
    ts = now()
    cur = conn.execute(
        "INSERT INTO comments (task_id, author_id, body, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, author_id, body, ts, ts),
    )
    return cur.lastrowid


def get_comment(conn, comment_id: int):
    return conn.execute("SELECT * FROM comments WHERE id = ?", (comment_id,)).fetchone()


def list_comments_for_task(conn, task_id: int):
    return conn.execute(
        "SELECT * FROM comments WHERE task_id = ? ORDER BY created_at ASC",
        (task_id,),
    ).fetchall()


def delete_comment(conn, comment_id: int):
    conn.execute("DELETE FROM comments WHERE id = ?", (comment_id,))


# -------------------------------------------------------------- notifications
def create_notification(conn, user_id: int, workspace_id: int, type_: str, message: str,
                         actor_id: int = None, task_id: int = None, comment_id: int = None):
    ts = now()
    cur = conn.execute(
        """INSERT INTO notifications
               (user_id, workspace_id, type, message, task_id, comment_id, actor_id, read_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
        (user_id, workspace_id, type_, message, task_id, comment_id, actor_id, ts),
    )
    return cur.lastrowid


def get_notification(conn, notification_id: int):
    return conn.execute("SELECT * FROM notifications WHERE id = ?", (notification_id,)).fetchone()


def list_notifications_for_user(conn, user_id: int, workspace_id: int,
                                 unread_only: bool = False, limit: int = 100):
    query = "SELECT * FROM notifications WHERE user_id = ? AND workspace_id = ?"
    params = [user_id, workspace_id]
    if unread_only:
        query += " AND read_at IS NULL"
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return conn.execute(query, params).fetchall()


def mark_notification_read(conn, notification_id: int):
    conn.execute("UPDATE notifications SET read_at = ? WHERE id = ?", (now(), notification_id))


def mark_all_notifications_read(conn, user_id: int, workspace_id: int) -> int:
    cur = conn.execute(
        "UPDATE notifications SET read_at = ? WHERE user_id = ? AND workspace_id = ? AND read_at IS NULL",
        (now(), user_id, workspace_id),
    )
    return cur.rowcount


def count_unread_notifications(conn, user_id: int, workspace_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM notifications WHERE user_id = ? AND workspace_id = ? AND read_at IS NULL",
        (user_id, workspace_id),
    ).fetchone()
    return row["n"]


def row_to_dict(row) -> dict:
    return dict(row) if row is not None else None
