"""Job-queue access for the RL inference daemon. Shares the Postgres database with the Node
Lambda (functions/api/lib/inference_db.mjs) but operates on the `inference_jobs` table — a
separate queue from the WRF `simulations` table so the two workers never claim each other's
jobs. Same connection discipline as worker/db.py: a lazily-(re)established connection that
retries each statement once on a dropped socket (serverless Postgres drops idle connections)."""
import json

import psycopg2
import psycopg2.extras

import config

# Status that means a worker is actively running the job (used to reclaim stale rows).
RUNNING_STATUSES = ("running",)

_DISCONNECT_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)

_conn = None


def _connect():
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set — cannot reach the job queue")
    conn = psycopg2.connect(
        config.DATABASE_URL,
        sslmode="require",
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
    try:
        return fn(_get_conn())
    except _DISCONNECT_ERRORS:
        _reset_conn()
        return fn(_get_conn())


def connect():
    """Eagerly establish the connection so the daemon fails fast at startup."""
    return _get_conn()


def claim_next_job():
    """Atomically pick the oldest queued job and flip it to 'running'. Returns the row dict
    (id, checkpoint_key, stats_key, mask_data_url, raw_features, sample, target_temp, place, date),
    or None if the queue is empty."""
    def op(conn):
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, checkpoint_key, stats_key, mask_data_url,
                       raw_features, sample, target_temp, place, date
                  FROM inference_jobs
                 WHERE status = 'queued'
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
                "UPDATE inference_jobs SET status = 'running', updated_at = now() WHERE id = %s",
                (row["id"],),
            )
        conn.commit()
        return row
    return _run(op)


def reclaim_stale_jobs(stale_minutes):
    """Requeue jobs stuck in 'running' whose worker died mid-run (crash/OOM/VM restart)."""
    def op(conn):
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE inference_jobs
                   SET status = 'queued', updated_at = now()
                 WHERE status = ANY(%s)
                   AND updated_at < now() - (%s || ' minutes')::interval
                """,
                (list(RUNNING_STATUSES), str(int(stale_minutes))),
            )
            n = cur.rowcount
        conn.commit()
        return n
    return _run(op)


def mark_failed(job_id, error):
    def op(conn):
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE inference_jobs SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
                (str(error)[:4000], job_id),
            )
        conn.commit()
    _run(op)


def mark_completed(job_id, result):
    def op(conn):
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE inference_jobs SET status = 'completed', result = %s, error = NULL, updated_at = now() WHERE id = %s",
                (json.dumps(result), job_id),
            )
        conn.commit()
    _run(op)
