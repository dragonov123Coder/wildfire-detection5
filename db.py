"""Local SQLite persistence for detection units and prediction runs.

Single file at instance/wildfire.db. WAL mode lets unit-connection threads,
Flask request threads, and the scheduler thread all read concurrently; writes
are additionally serialized through _WRITE_LOCK since sqlite3 connections
aren't safe for concurrent writers even with check_same_thread=False.
"""
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / 'instance' / 'wildfire.db'

_WRITE_LOCK = threading.Lock()
_conn = None
_conn_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    global _conn
    with _conn_lock:
        if _conn is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            _conn.execute('PRAGMA journal_mode=WAL')
            _conn.row_factory = sqlite3.Row
        return _conn


def init_db():
    conn = get_connection()
    with _WRITE_LOCK:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS units (
                unit_id     TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                lat         REAL,
                lon         REAL,
                created_at  TEXT NOT NULL,
                last_seen   TEXT,
                status      TEXT NOT NULL DEFAULT 'awaiting_connection'
            );

            CREATE TABLE IF NOT EXISTS prediction_runs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date            TEXT NOT NULL,
                target_date         TEXT NOT NULL,
                csv_path            TEXT NOT NULL,
                max_risk_score_raw  REAL,
                created_at          TEXT NOT NULL,
                UNIQUE(run_date, target_date)
            );
            CREATE INDEX IF NOT EXISTS idx_prediction_runs_target ON prediction_runs(target_date);

            CREATE TABLE IF NOT EXISTS notifications (
                run_date            TEXT PRIMARY KEY,
                sent_at             TEXT NOT NULL,
                max_risk_score_raw  REAL NOT NULL,
                target_date         TEXT NOT NULL
            );
            """
        )
        conn.commit()


# --- units ---------------------------------------------------------------

def next_unit_id() -> str:
    conn = get_connection()
    row = conn.execute(
        "SELECT MAX(CAST(SUBSTR(unit_id, 6) AS INTEGER)) AS n FROM units WHERE unit_id LIKE 'unit-%'"
    ).fetchone()
    n = (row['n'] or 0) + 1
    return f"unit-{n:03d}"


def create_unit(name: str, lat, lon, unit_id: str = None) -> dict:
    """Create a unit. If unit_id is given (e.g. a Pi auto-registering itself
    with an id it already carries), use it as-is instead of generating one."""
    conn = get_connection()
    with _WRITE_LOCK:
        uid = unit_id or next_unit_id()
        created_at = _now()
        conn.execute(
            "INSERT INTO units (unit_id, name, lat, lon, created_at, status) VALUES (?, ?, ?, ?, ?, 'awaiting_connection')",
            (uid, name, lat, lon, created_at),
        )
        conn.commit()
    return get_unit(uid)


def list_units() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM units ORDER BY created_at ASC").fetchall()
    return [dict(r) for r in rows]


def get_unit(unit_id: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM units WHERE unit_id = ?", (unit_id,)).fetchone()
    return dict(row) if row else None


def touch_unit_last_seen(unit_id: str):
    conn = get_connection()
    with _WRITE_LOCK:
        conn.execute("UPDATE units SET last_seen = ? WHERE unit_id = ?", (_now(), unit_id))
        conn.commit()


def set_unit_status(unit_id: str, status: str):
    conn = get_connection()
    with _WRITE_LOCK:
        conn.execute("UPDATE units SET status = ? WHERE unit_id = ?", (status, unit_id))
        conn.commit()


def set_unit_location_if_unset(unit_id: str, lat, lon):
    """Fix a unit's central location from its first GPS fix. No-op if the
    unit already has a location (deploy-time click, or an earlier fix)."""
    if lat is None or lon is None:
        return
    conn = get_connection()
    with _WRITE_LOCK:
        conn.execute(
            "UPDATE units SET lat = ?, lon = ? WHERE unit_id = ? AND lat IS NULL AND lon IS NULL",
            (lat, lon, unit_id),
        )
        conn.commit()


# --- prediction runs -------------------------------------------------------

def record_prediction_run(run_date: str, target_date: str, csv_path: str, max_risk_score_raw):
    conn = get_connection()
    with _WRITE_LOCK:
        conn.execute(
            """INSERT INTO prediction_runs (run_date, target_date, csv_path, max_risk_score_raw, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(run_date, target_date) DO UPDATE SET
                   csv_path=excluded.csv_path,
                   max_risk_score_raw=excluded.max_risk_score_raw,
                   created_at=excluded.created_at""",
            (run_date, target_date, csv_path, max_risk_score_raw, _now()),
        )
        conn.commit()


def latest_prediction_runs() -> list:
    """Rows for the most recent run_date on record."""
    conn = get_connection()
    row = conn.execute("SELECT MAX(run_date) AS d FROM prediction_runs").fetchone()
    if not row or not row['d']:
        return []
    rows = conn.execute(
        "SELECT * FROM prediction_runs WHERE run_date = ? ORDER BY target_date ASC", (row['d'],)
    ).fetchall()
    return [dict(r) for r in rows]


def csv_path_for_target(target_date: str):
    """Most recently-run CSV path that covers this target date, if any."""
    conn = get_connection()
    row = conn.execute(
        "SELECT csv_path FROM prediction_runs WHERE target_date = ? ORDER BY run_date DESC LIMIT 1",
        (target_date,),
    ).fetchone()
    return row['csv_path'] if row else None


def is_first_prediction_run() -> bool:
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) AS n FROM prediction_runs").fetchone()
    return row['n'] == 0


def notification_already_sent(run_date: str) -> bool:
    conn = get_connection()
    row = conn.execute("SELECT 1 FROM notifications WHERE run_date = ?", (run_date,)).fetchone()
    return row is not None


def mark_notification_sent(run_date: str, max_risk_score_raw: float, target_date: str):
    conn = get_connection()
    with _WRITE_LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO notifications (run_date, sent_at, max_risk_score_raw, target_date) VALUES (?, ?, ?, ?)",
            (run_date, _now(), max_risk_score_raw, target_date),
        )
        conn.commit()
