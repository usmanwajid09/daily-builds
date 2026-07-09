"""SQLite schema + connection helpers for the dev-collab-platform.

Data model
----------
workspaces      -- one row per organization/team
users           -- global user identity (an email can belong to many workspaces)
memberships     -- join table: user_id + workspace_id -> role
projects        -- belong to exactly one workspace
tasks           -- belong to exactly one project; status is the task-board column

Every workspace-scoped query filters by workspace_id, and that
workspace_id always comes from the verified JWT on the request/WS
connection -- never from a client-supplied field -- so there is no
request parameter that lets one workspace read or write another's data.
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
    project_id INTEGER NOT NULL REFERENCES projects(id),
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'todo' CHECK(status IN ('todo', 'in_progress', 'done')),
    position INTEGER NOT NULL DEFAULT 0,
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memberships_user ON memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_memberships_workspace ON memberships(workspace_id);
CREATE INDEX IF NOT EXISTS idx_projects_workspace ON projects(workspace_id);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
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
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))


def row_to_dict(row) -> dict:
    return dict(row) if row is not None else None
