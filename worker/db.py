"""Job-queue access. The worker shares the `simulations` table with the Node Lambda
(functions/api/lib/db.mjs). Jobs are claimed atomically with SELECT ... FOR UPDATE
SKIP LOCKED so multiple workers never grab the same row.

A single WRF stage (baseline/candidate wrf.exe) runs for minutes with no DB traffic, and
serverless Postgres (Neon) drops idle connections — so the next status write would hit a
dead socket ("SSL connection has been closed unexpectedly"). To survive that, this module
owns a lazily-(re)established connection and retries each statement once on a connection
error, reconnecting first. TCP keepalives reduce, but cannot eliminate, server-side idle
drops, so the retry is the real safety net."""
import json
import psycopg2
import psycopg2.extras

import config

# Worker-side status vocabulary (must match STATUSES in functions/api/lib/schema.mjs).
RUNNING_STATUSES = (
    "preprocessing",
    "initializing",
    "running_baseline",
    "running_candidate",
    "postprocessing",
)

# psycopg2 errors that mean the socket is gone and a reconnect is worth trying.
_DISCONNECT_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)

_conn = None


def _connect():
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set — cannot reach the job queue")
    conn = psycopg2.connect(
        config.DATABASE_URL,
        sslmode="require",
        # Nudge the OS to detect a dead peer instead of blocking forever.
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    conn.autocommit = False
    return conn


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = _connect()
    return _conn


def _reset_conn():
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    _conn = None


def _run(fn):
    """Run `fn(conn)` with one reconnect-and-retry on a dropped connection. `fn` must be
    idempotent at the statement level (a single SELECT/UPDATE), which all callers here are."""
    try:
        return fn(_get_conn())
    except _DISCONNECT_ERRORS:
        _reset_conn()
        return fn(_get_conn())


def connect():
    """Eagerly establish the connection so the daemon fails fast at startup if the queue
    is unreachable. Subsequent operations self-heal via _run."""
    return _get_conn()


def claim_next_job():
    """Atomically pick the oldest queued job and flip it to 'preprocessing'.
    Returns the row dict, or None if the queue is empty."""
    def op(conn):
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, scenario
                  FROM simulations
                 WHERE status = 'created'
                 ORDER BY created_at ASC
                 FOR UPDATE SKIP LOCKED
                 LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return None
            cur.execute(
                "UPDATE simulations SET status = 'preprocessing', updated_at = now() WHERE id = %s",
                (row["id"],),
            )
        conn.commit()
        return row
    return _run(op)


def reclaim_stale_jobs(stale_minutes):
    """Requeue jobs stuck in a running state whose worker died mid-run.

    A healthy worker bumps updated_at between stages, so a row that has sat in a
    running status past `stale_minutes` without an update has lost its worker (VM
    restart, OOM, killed process, dropped DB connection mid-run). Flip it back to
    'created' so it is retried instead of spinning forever in the UI. `stale_minutes`
    must exceed the longest single WRF stage on the clamped budget. Returns the count."""
    def op(conn):
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE simulations
                   SET status = 'created', updated_at = now()
                 WHERE status = ANY(%s)
                   AND updated_at < now() - (%s || ' minutes')::interval
                """,
                (list(RUNNING_STATUSES), str(int(stale_minutes))),
            )
            n = cur.rowcount
        conn.commit()
        return n
    return _run(op)


def set_status(job_id, status):
    def op(conn):
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE simulations SET status = %s, updated_at = now() WHERE id = %s",
                (status, job_id),
            )
        conn.commit()
    _run(op)


def mark_failed(job_id, error):
    """The ONLY honest terminal state when WRF cannot run. Never write 'completed'
    unless a real wrfout was produced and post-processed."""
    def op(conn):
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE simulations SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
                (str(error)[:4000], job_id),
            )
        conn.commit()
    _run(op)


def mark_completed(job_id, result):
    def op(conn):
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE simulations SET status = 'completed', result = %s, updated_at = now() WHERE id = %s",
                (json.dumps(result), job_id),
            )
        conn.commit()
    _run(op)
