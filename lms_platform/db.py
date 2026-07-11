"""SQLite schema + data-access helpers for the LMS platform.

Dependency-free (stdlib sqlite3), following the same pattern as
dev_collab_platform/app/db.py and saas_starter: thin functions taking a
connection and plain arguments, returning sqlite3.Row objects that
route handlers convert to dicts with row_to_dict(). One process-wide
connection is opened by the app factory; writes go through the
`transaction()` context manager so a handler's multi-statement writes
either all land or all roll back.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('instructor', 'student')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    instructor_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content_type TEXT NOT NULL CHECK (content_type IN ('text', 'video_url', 'file')),
    content TEXT NOT NULL DEFAULT '',
    order_index INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (course_id, order_index)
);

CREATE TABLE IF NOT EXISTS enrollments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    enrolled_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (student_id, course_id)
);

CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    enrollment_id INTEGER NOT NULL REFERENCES enrollments(id) ON DELETE CASCADE,
    lesson_id INTEGER NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    completed_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (enrollment_id, lesson_id)
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Commit on success, rollback on any exception."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


# -- users -------------------------------------------------------------

def create_user(conn, email: str, password_hash: str, role: str) -> int:
    cur = conn.execute(
        "INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
        (email, password_hash, role),
    )
    return cur.lastrowid


def get_user(conn, user_id: int):
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_email(conn, email: str):
    return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


# -- courses -------------------------------------------------------------

def create_course(conn, title: str, description: str, instructor_id: int) -> int:
    cur = conn.execute(
        "INSERT INTO courses (title, description, instructor_id) VALUES (?, ?, ?)",
        (title, description, instructor_id),
    )
    return cur.lastrowid


def get_course(conn, course_id: int):
    return conn.execute("SELECT * FROM courses WHERE id = ?", (course_id,)).fetchone()


def list_courses(conn, instructor_id: int | None = None):
    if instructor_id is not None:
        return conn.execute(
            "SELECT * FROM courses WHERE instructor_id = ? ORDER BY id", (instructor_id,)
        ).fetchall()
    return conn.execute("SELECT * FROM courses ORDER BY id").fetchall()


def update_course(conn, course_id: int, **fields) -> None:
    if not fields:
        return
    allowed = {"title", "description"}
    set_clause = ", ".join(f"{k} = ?" for k in fields if k in allowed)
    values = [v for k, v in fields.items() if k in allowed]
    if not set_clause:
        return
    values.append(course_id)
    conn.execute(f"UPDATE courses SET {set_clause} WHERE id = ?", values)


def delete_course(conn, course_id: int) -> None:
    conn.execute("DELETE FROM courses WHERE id = ?", (course_id,))


# -- lessons -------------------------------------------------------------

def create_lesson(conn, course_id: int, title: str, content_type: str, content: str,
                   order_index: int) -> int:
    cur = conn.execute(
        "INSERT INTO lessons (course_id, title, content_type, content, order_index) "
        "VALUES (?, ?, ?, ?, ?)",
        (course_id, title, content_type, content, order_index),
    )
    return cur.lastrowid


def get_lesson(conn, lesson_id: int):
    return conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()


def list_lessons_for_course(conn, course_id: int):
    return conn.execute(
        "SELECT * FROM lessons WHERE course_id = ? ORDER BY order_index", (course_id,)
    ).fetchall()


def next_lesson_order_index(conn, course_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(order_index), 0) + 1 AS next FROM lessons WHERE course_id = ?",
        (course_id,),
    ).fetchone()
    return row["next"]


def update_lesson(conn, lesson_id: int, **fields) -> None:
    if not fields:
        return
    allowed = {"title", "content_type", "content", "order_index"}
    set_clause = ", ".join(f"{k} = ?" for k in fields if k in allowed)
    values = [v for k, v in fields.items() if k in allowed]
    if not set_clause:
        return
    values.append(lesson_id)
    conn.execute(f"UPDATE lessons SET {set_clause} WHERE id = ?", values)


def delete_lesson(conn, lesson_id: int) -> None:
    conn.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))


# -- enrollments -------------------------------------------------------------

def create_enrollment(conn, student_id: int, course_id: int) -> int:
    cur = conn.execute(
        "INSERT INTO enrollments (student_id, course_id) VALUES (?, ?)",
        (student_id, course_id),
    )
    return cur.lastrowid


def get_enrollment(conn, student_id: int, course_id: int):
    return conn.execute(
        "SELECT * FROM enrollments WHERE student_id = ? AND course_id = ?",
        (student_id, course_id),
    ).fetchone()


def get_enrollment_by_id(conn, enrollment_id: int):
    return conn.execute("SELECT * FROM enrollments WHERE id = ?", (enrollment_id,)).fetchone()


def list_enrollments_for_student(conn, student_id: int):
    return conn.execute(
        "SELECT * FROM enrollments WHERE student_id = ? ORDER BY id", (student_id,)
    ).fetchall()


# -- progress -------------------------------------------------------------

def mark_lesson_complete(conn, enrollment_id: int, lesson_id: int) -> None:
    """Idempotent: completing an already-completed lesson is a no-op,
    not an integrity error (INSERT OR IGNORE against the UNIQUE
    constraint on (enrollment_id, lesson_id))."""
    conn.execute(
        "INSERT OR IGNORE INTO progress (enrollment_id, lesson_id) VALUES (?, ?)",
        (enrollment_id, lesson_id),
    )


def list_completed_lesson_ids(conn, enrollment_id: int) -> set[int]:
    rows = conn.execute(
        "SELECT lesson_id FROM progress WHERE enrollment_id = ?", (enrollment_id,)
    ).fetchall()
    return {r["lesson_id"] for r in rows}
