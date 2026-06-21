/**
 * Postgres access for the RL inference job queue. Separate table from `simulations`
 * (lib/db.mjs) so the WRF worker never claims RL jobs and vice-versa. Same pool guidance:
 * small max, long idle, reused across warm Lambda invocations.
 *
 * The Lambda only enqueues + reads. A Python torch daemon (worker/rl_inference_worker.py)
 * claims `status='queued'` rows, runs the model, and writes `result`/`status`.
 */
import { getPool } from './db.mjs';

let inferenceSchemaReady = false;
export async function ensureInferenceSchema() {
  const db = getPool();
  if (!db) throw new Error('DATABASE_URL not configured');
  if (inferenceSchemaReady) return;
  await db.query(`
    CREATE TABLE IF NOT EXISTS inference_jobs (
      id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      status        text NOT NULL DEFAULT 'queued',
      checkpoint_key text NOT NULL,
      stats_key     text,
      mask_data_url text NOT NULL,
      raw_features  jsonb NOT NULL,
      target_temp   double precision,
      place         text,
      date          text,
      result        jsonb,
      error         text,
      created_at    timestamptz NOT NULL DEFAULT now(),
      updated_at    timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS inference_jobs_status_created_idx
      ON inference_jobs (status, created_at);
  `);
  inferenceSchemaReady = true;
}

export async function createInferenceJob({
  checkpoint_key,
  stats_key,
  mask_data_url,
  raw_features,
  target_temp,
  place,
  date,
}) {
  const db = getPool();
  const { rows } = await db.query(
    `INSERT INTO inference_jobs
       (status, checkpoint_key, stats_key, mask_data_url, raw_features, target_temp, place, date)
     VALUES ('queued', $1, $2, $3, $4, $5, $6, $7)
     RETURNING id, status, created_at`,
    [
      checkpoint_key,
      stats_key || null,
      mask_data_url,
      JSON.stringify(raw_features),
      Number.isFinite(Number(target_temp)) ? Number(target_temp) : null,
      place || null,
      date || null,
    ]
  );
  return rows[0];
}

export async function getInferenceJob(id) {
  const db = getPool();
  const { rows } = await db.query(
    `SELECT id, status, result, error, created_at, updated_at FROM inference_jobs WHERE id = $1`,
    [id]
  );
  return rows[0] || null;
}
