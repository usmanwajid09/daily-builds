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

CREATE TABLE IF NOT EXISTS quizzes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    passing_score REAL NOT NULL DEFAULT 70.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS quiz_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    points REAL NOT NULL DEFAULT 1.0,
    UNIQUE (quiz_id, order_index)
);

CREATE TABLE IF NOT EXISTS quiz_options (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
    option_text TEXT NOT NULL,
    is_correct INTEGER NOT NULL DEFAULT 0 CHECK (is_correct IN (0, 1)),
    order_index INTEGER NOT NULL,
    UNIQUE (question_id, order_index)
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    student_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    score REAL NOT NULL,
    total_points REAL NOT NULL,
    percent REAL NOT NULL,
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    submitted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS quiz_attempt_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL REFERENCES quiz_attempts(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
    option_id INTEGER REFERENCES quiz_options(id) ON DELETE SET NULL,
    correct INTEGER NOT NULL CHECK (correct IN (0, 1))
);

CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    max_points REAL NOT NULL DEFAULT 100.0,
    due_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS assignment_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
    student_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
    grade REAL,
    feedback TEXT,
    graded_at TEXT,
    UNIQUE (assignment_id, student_id)
);

CREATE TABLE IF NOT EXISTS certificates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    verification_code TEXT NOT NULL UNIQUE,
    issued_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (student_id, course_id)
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


# -- quizzes -------------------------------------------------------------

def create_quiz(conn, course_id: int, title: str, passing_score: float) -> int:
    cur = conn.execute(
        "INSERT INTO quizzes (course_id, title, passing_score) VALUES (?, ?, ?)",
        (course_id, title, passing_score),
    )
    return cur.lastrowid


def get_quiz(conn, quiz_id: int):
    return conn.execute("SELECT * FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()


def list_quizzes_for_course(conn, course_id: int):
    return conn.execute(
        "SELECT * FROM quizzes WHERE course_id = ? ORDER BY id", (course_id,)
    ).fetchall()


def delete_quiz(conn, quiz_id: int) -> None:
    conn.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))


def create_quiz_question(conn, quiz_id: int, question_text: str, order_index: int,
                          points: float) -> int:
    cur = conn.execute(
        "INSERT INTO quiz_questions (quiz_id, question_text, order_index, points) "
        "VALUES (?, ?, ?, ?)",
        (quiz_id, question_text, order_index, points),
    )
    return cur.lastrowid


def list_questions_for_quiz(conn, quiz_id: int):
    return conn.execute(
        "SELECT * FROM quiz_questions WHERE quiz_id = ? ORDER BY order_index", (quiz_id,)
    ).fetchall()


def get_question(conn, question_id: int):
    return conn.execute(
        "SELECT * FROM quiz_questions WHERE id = ?", (question_id,)
    ).fetchone()


def create_quiz_option(conn, question_id: int, option_text: str, is_correct: bool,
                        order_index: int) -> int:
    cur = conn.execute(
        "INSERT INTO quiz_options (question_id, option_text, is_correct, order_index) "
        "VALUES (?, ?, ?, ?)",
        (question_id, option_text, 1 if is_correct else 0, order_index),
    )
    return cur.lastrowid


def list_options_for_question(conn, question_id: int):
    return conn.execute(
        "SELECT * FROM quiz_options WHERE question_id = ? ORDER BY order_index",
        (question_id,),
    ).fetchall()


def get_option(conn, option_id: int):
    return conn.execute("SELECT * FROM quiz_options WHERE id = ?", (option_id,)).fetchone()


def create_quiz_attempt(conn, quiz_id: int, student_id: int, score: float,
                         total_points: float, percent: float, passed: bool) -> int:
    cur = conn.execute(
        "INSERT INTO quiz_attempts (quiz_id, student_id, score, total_points, percent, passed) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (quiz_id, student_id, score, total_points, percent, 1 if passed else 0),
    )
    return cur.lastrowid


def record_attempt_answer(conn, attempt_id: int, question_id: int, option_id: int | None,
                           correct: bool) -> None:
    conn.execute(
        "INSERT INTO quiz_attempt_answers (attempt_id, question_id, option_id, correct) "
        "VALUES (?, ?, ?, ?)",
        (attempt_id, question_id, option_id, 1 if correct else 0),
    )


def list_attempts_for_quiz(conn, quiz_id: int):
    return conn.execute(
        "SELECT * FROM quiz_attempts WHERE quiz_id = ? ORDER BY submitted_at", (quiz_id,)
    ).fetchall()


def list_attempts_for_student(conn, student_id: int, quiz_id: int | None = None):
    if quiz_id is not None:
        return conn.execute(
            "SELECT * FROM quiz_attempts WHERE student_id = ? AND quiz_id = ? "
            "ORDER BY submitted_at",
            (student_id, quiz_id),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM quiz_attempts WHERE student_id = ? ORDER BY submitted_at",
        (student_id,),
    ).fetchall()


def best_attempt_for_student(conn, student_id: int, quiz_id: int):
    """Highest-percent attempt (ties broken by most recent). None if the
    student has never attempted this quiz."""
    return conn.execute(
        "SELECT * FROM quiz_attempts WHERE student_id = ? AND quiz_id = ? "
        "ORDER BY percent DESC, submitted_at DESC LIMIT 1",
        (student_id, quiz_id),
    ).fetchone()


# -- assignments -------------------------------------------------------------

def create_assignment(conn, course_id: int, title: str, description: str,
                       max_points: float, due_at: str | None) -> int:
    cur = conn.execute(
        "INSERT INTO assignments (course_id, title, description, max_points, due_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (course_id, title, description, max_points, due_at),
    )
    return cur.lastrowid


def get_assignment(conn, assignment_id: int):
    return conn.execute(
        "SELECT * FROM assignments WHERE id = ?", (assignment_id,)
    ).fetchone()


def list_assignments_for_course(conn, course_id: int):
    return conn.execute(
        "SELECT * FROM assignments WHERE course_id = ? ORDER BY id", (course_id,)
    ).fetchall()


def delete_assignment(conn, assignment_id: int) -> None:
    conn.execute("DELETE FROM assignments WHERE id = ?", (assignment_id,))


def upsert_submission(conn, assignment_id: int, student_id: int, content: str) -> int:
    """Create a submission, or overwrite an existing one (resetting any
    prior grade -- a resubmission always needs re-grading)."""
    existing = get_submission(conn, assignment_id, student_id)
    if existing is None:
        cur = conn.execute(
            "INSERT INTO assignment_submissions (assignment_id, student_id, content) "
            "VALUES (?, ?, ?)",
            (assignment_id, student_id, content),
        )
        return cur.lastrowid
    conn.execute(
        "UPDATE assignment_submissions SET content = ?, submitted_at = datetime('now'), "
        "grade = NULL, feedback = NULL, graded_at = NULL WHERE id = ?",
        (content, existing["id"]),
    )
    return existing["id"]


def get_submission(conn, assignment_id: int, student_id: int):
    return conn.execute(
        "SELECT * FROM assignment_submissions WHERE assignment_id = ? AND student_id = ?",
        (assignment_id, student_id),
    ).fetchone()


def get_submission_by_id(conn, submission_id: int):
    return conn.execute(
        "SELECT * FROM assignment_submissions WHERE id = ?", (submission_id,)
    ).fetchone()


def list_submissions_for_assignment(conn, assignment_id: int):
    return conn.execute(
        "SELECT * FROM assignment_submissions WHERE assignment_id = ? ORDER BY submitted_at",
        (assignment_id,),
    ).fetchall()


def grade_submission(conn, submission_id: int, grade: float, feedback: str) -> None:
    conn.execute(
        "UPDATE assignment_submissions SET grade = ?, feedback = ?, "
        "graded_at = datetime('now') WHERE id = ?",
        (grade, feedback, submission_id),
    )


# -- certificates -------------------------------------------------------------

def create_certificate(conn, student_id: int, course_id: int, verification_code: str) -> int:
    cur = conn.execute(
        "INSERT INTO certificates (student_id, course_id, verification_code) "
        "VALUES (?, ?, ?)",
        (student_id, course_id, verification_code),
    )
    return cur.lastrowid


def get_certificate_for_course(conn, student_id: int, course_id: int):
    return conn.execute(
        "SELECT * FROM certificates WHERE student_id = ? AND course_id = ?",
        (student_id, course_id),
    ).fetchone()


def get_certificate_by_code(conn, verification_code: str):
    return conn.execute(
        "SELECT * FROM certificates WHERE verification_code = ?", (verification_code,)
    ).fetchone()


def list_certificates_for_student(conn, student_id: int):
    return conn.execute(
        "SELECT * FROM certificates WHERE student_id = ? ORDER BY issued_at",
        (student_id,),
    ).fetchall()
