"""SQLite persistence for the Medium Agent.

This is the agent's *memory*. Level 1 was stateless - every question started from
nothing. Here the agent can look a learner up, write a plan, come back a week
later and record progress against that same plan.

Three tables:

  learners        one row per employee - the profile the agent reasons over
  learning_plans  one row per employee - the CURRENT plan, stored as JSON
  completions     append-only log - one row every time a course is completed

Everything is plain sqlite3 from the standard library. No ORM, no migrations,
no external database server.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

# Override with LEARNER_DB=... if you want a throwaway database for a demo.
DB_PATH = os.getenv("LEARNER_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "learner.db"))


def _now() -> str:
    """UTC timestamp as an ISO-8601 string. SQLite has no native date type."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    """Open a connection with row access by column name, commit, and always close.

    Written as a context manager rather than returning the connection, because
    sqlite3's own `with conn:` block commits the transaction but does NOT close the
    handle. The connection then lives until garbage collection, which on Windows
    keeps a lock on the .db file and makes it undeletable - the eval harness caught
    exactly that. Closing here means every caller is clean by construction.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def key_for_email(email: str) -> str:
    """The learner's key, derived from their email. The email IS the identity.

    Normalised so "Deepak@LevelShift.com " and "deepak@levelshift.com" resolve to one
    learner with one plan, rather than two records that each think they are the only
    one. Angle brackets get stripped because mail clients paste them.
    """
    return (email or "").strip().strip("<>").strip().lower()


# Columns added after the first version of the schema. Databases created before a
# column existed get it via ALTER TABLE - the cheap version of a migration, which is
# all a single-file SQLite database needs.
LATER_LEARNER_COLUMNS = {
    "weeks_completed": "REAL NOT NULL DEFAULT 0",
}


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(learners)")}
    for column, ddl in LATER_LEARNER_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE learners ADD COLUMN {column} {ddl}")


def init_db() -> None:
    """Create the four tables if they do not exist yet. Safe to call every run."""
    with connect() as conn:
        conn.executescript(
            """
            -- employee_id holds the learner's EMAIL, normalised. The email is the
            -- identity: one address, one record, one plan, one progress history.
            CREATE TABLE IF NOT EXISTS learners (
                employee_id      TEXT PRIMARY KEY,
                name             TEXT,
                current_role     TEXT,
                experience_years REAL,
                current_skills   TEXT NOT NULL DEFAULT '[]',  -- JSON array
                target_role      TEXT,
                hours_per_week   REAL,
                timeline_months  REAL,
                weeks_completed  REAL NOT NULL DEFAULT 0,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS learning_plans (
                employee_id  TEXT PRIMARY KEY,
                plan_json    TEXT NOT NULL,
                version      INTEGER NOT NULL DEFAULT 1,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES learners (employee_id)
            );

            CREATE TABLE IF NOT EXISTS completions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id   TEXT NOT NULL,
                course_id     TEXT NOT NULL,
                was_in_plan   INTEGER NOT NULL,   -- 0/1, so off-plan learning is visible
                completed_at  TEXT NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES learners (employee_id)
            );

            -- The course catalogue. There is no knowledge_base.json any more: the
            -- model proposes courses via search_courses and they are cached here so
            -- ids stay stable, plans stay checkable, and completions can be
            -- validated in a later process that never called search_courses.
            CREATE TABLE IF NOT EXISTS courses (
                id             TEXT PRIMARY KEY,
                title          TEXT NOT NULL,
                provider       TEXT,
                topic          TEXT,
                level          TEXT,
                duration_hours REAL,
                prerequisites  TEXT NOT NULL DEFAULT '[]',  -- JSON array of course ids
                skills_taught  TEXT NOT NULL DEFAULT '[]',  -- JSON array
                course_link    TEXT,
                content        TEXT,
                source         TEXT NOT NULL DEFAULT 'model-generated',
                created_at     TEXT NOT NULL
            );
            """
        )
        _add_missing_columns(conn)


# --------------------------------------------------------------------------
# learners
# --------------------------------------------------------------------------

def get_learner(employee_id: str) -> dict | None:
    """Return the learner row as a dict, or None if there is no such employee."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM learners WHERE employee_id = ?", (employee_id,)
        ).fetchone()

    if row is None:
        return None

    learner = dict(row)
    learner["current_skills"] = json.loads(learner["current_skills"] or "[]")
    return learner


def create_learner(employee_id: str, **fields) -> dict:
    """Insert a new learner row with whatever fields we know so far."""
    now = _now()
    skills = json.dumps(fields.get("current_skills") or [])

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO learners (employee_id, name, current_role, experience_years,
                                  current_skills, target_role, hours_per_week,
                                  timeline_months, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                employee_id,
                fields.get("name"),
                fields.get("current_role"),
                fields.get("experience_years"),
                skills,
                fields.get("target_role"),
                fields.get("hours_per_week"),
                fields.get("timeline_months"),
                now,
                now,
            ),
        )

    return get_learner(employee_id)


def update_learner(employee_id: str, **fields) -> dict | None:
    """Patch only the fields that were passed in. Unknown keys are ignored."""
    allowed = {
        "name", "current_role", "experience_years", "current_skills",
        "target_role", "hours_per_week", "timeline_months", "weeks_completed",
    }
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return get_learner(employee_id)

    if "current_skills" in updates:
        updates["current_skills"] = json.dumps(updates["current_skills"])

    assignments = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [_now(), employee_id]

    with connect() as conn:
        conn.execute(
            f"UPDATE learners SET {assignments}, updated_at = ? WHERE employee_id = ?",
            values,
        )

    return get_learner(employee_id)


# --------------------------------------------------------------------------
# learning_plans
# --------------------------------------------------------------------------

def save_plan(employee_id: str, plan: dict) -> dict:
    """Insert or overwrite this learner's current plan. Bumps the version counter."""
    now = _now()
    plan_json = json.dumps(plan, indent=2)

    with connect() as conn:
        existing = conn.execute(
            "SELECT version, created_at FROM learning_plans WHERE employee_id = ?",
            (employee_id,),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO learning_plans (employee_id, plan_json, version,
                                            created_at, updated_at)
                VALUES (?, ?, 1, ?, ?)
                """,
                (employee_id, plan_json, now, now),
            )
            version = 1
        else:
            version = existing["version"] + 1
            conn.execute(
                """
                UPDATE learning_plans
                   SET plan_json = ?, version = ?, updated_at = ?
                 WHERE employee_id = ?
                """,
                (plan_json, version, now, employee_id),
            )

    return {"employee_id": employee_id, "version": version, "updated_at": now}


def get_plan_record(employee_id: str) -> dict | None:
    """The current plan plus its version and timestamps, or None if there is no plan.

    Replanning needs the version and updated_at, not just the plan: "version 3,
    revised on 2026-08-07" is what tells a learner their path was adjusted.
    """
    with connect() as conn:
        row = conn.execute(
            """
            SELECT plan_json, version, created_at, updated_at
              FROM learning_plans
             WHERE employee_id = ?
            """,
            (employee_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "plan": json.loads(row["plan_json"]),
        "version": row["version"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_plan(employee_id: str) -> dict | None:
    """Return the current plan dict, or None if this learner has no plan yet."""
    record = get_plan_record(employee_id)
    return record["plan"] if record else None


def plan_course_ids(employee_id: str) -> list[str]:
    """Flatten every course_id across every week of the current plan."""
    plan = get_plan(employee_id)
    if not plan:
        return []

    ids: list[str] = []
    for week in plan.get("weekly_plan", []):
        for course_id in week.get("courses", []):
            if course_id not in ids:
                ids.append(course_id)
    return ids


# --------------------------------------------------------------------------
# completions
# --------------------------------------------------------------------------

def log_completion(employee_id: str, course_id: str, was_in_plan: bool) -> None:
    """Append one completion event. Duplicates are allowed - it is a log."""
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO completions (employee_id, course_id, was_in_plan, completed_at)
            VALUES (?, ?, ?, ?)
            """,
            (employee_id, course_id, 1 if was_in_plan else 0, _now()),
        )


def completed_course_ids(employee_id: str) -> list[str]:
    """Distinct course ids this learner has completed, oldest first."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT course_id, MIN(completed_at) AS first_completed
              FROM completions
             WHERE employee_id = ?
             GROUP BY course_id
             ORDER BY first_completed
            """,
            (employee_id,),
        ).fetchall()

    return [row["course_id"] for row in rows]


def completion_log(employee_id: str) -> list[dict]:
    """Every completion event for this learner, oldest first.

    completed_course_ids() collapses repeats; this keeps the raw history, which is
    what a progress report shows ("3 courses, last one on 2026-08-07").
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT course_id, was_in_plan, completed_at
              FROM completions
             WHERE employee_id = ?
             ORDER BY completed_at, id
            """,
            (employee_id,),
        ).fetchall()

    return [
        {
            "course_id": row["course_id"],
            "was_in_plan": bool(row["was_in_plan"]),
            "completed_at": row["completed_at"],
        }
        for row in rows
    ]


# --------------------------------------------------------------------------
# courses (the model-generated catalogue)
# --------------------------------------------------------------------------

def _row_to_course(row: sqlite3.Row) -> dict:
    course = dict(row)
    course["prerequisites"] = json.loads(course["prerequisites"] or "[]")
    course["skills_taught"] = json.loads(course["skills_taught"] or "[]")
    return course


def upsert_courses(courses: list[dict]) -> int:
    """Insert or refresh courses the model proposed. Returns how many were written."""
    now = _now()
    written = 0

    with connect() as conn:
        for course in courses:
            course_id = str(course.get("id") or "").strip().upper()
            title = str(course.get("title") or "").strip()
            if not course_id or not title:
                continue

            try:
                hours = float(course.get("duration_hours") or 0)
            except (TypeError, ValueError):
                hours = 0.0

            conn.execute(
                """
                INSERT INTO courses (id, title, provider, topic, level, duration_hours,
                                     prerequisites, skills_taught, course_link, content,
                                     source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title          = excluded.title,
                    provider       = excluded.provider,
                    topic          = excluded.topic,
                    level          = excluded.level,
                    duration_hours = excluded.duration_hours,
                    prerequisites  = excluded.prerequisites,
                    skills_taught  = excluded.skills_taught,
                    course_link    = excluded.course_link,
                    content        = excluded.content
                """,
                (
                    course_id,
                    title,
                    course.get("provider"),
                    course.get("topic"),
                    course.get("level"),
                    hours,
                    json.dumps([str(p).strip().upper() for p in course.get("prerequisites") or []]),
                    json.dumps(course.get("skills_taught") or []),
                    course.get("course_link"),
                    course.get("content"),
                    course.get("source") or "model-generated",
                    now,
                ),
            )
            written += 1

    return written


def all_courses() -> list[dict]:
    """Every course cached so far. This is what "the catalogue" now means."""
    with connect() as conn:
        rows = conn.execute("SELECT * FROM courses ORDER BY id").fetchall()
    return [_row_to_course(row) for row in rows]


def get_course(course_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM courses WHERE id = ?", ((course_id or "").strip().upper(),)
        ).fetchone()
    return _row_to_course(row) if row else None


if __name__ == "__main__":
    init_db()
    print(f"Initialised database at {DB_PATH}")
