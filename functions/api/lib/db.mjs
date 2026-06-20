/**
 * Postgres access for the simulation job queue. Connection pooling per OpenKBS guidance
 * (small max, long idle) so the Lambda reuses connections across warm invocations.
 */
import pg from 'pg';

let pool;
export function getPool() {
  if (!pool && process.env.DATABASE_URL) {
    pool = new pg.Pool({
      connectionString: process.env.DATABASE_URL,
      ssl: { rejectUnauthorized: true },
      max: 3,
      idleTimeoutMillis: 60_000,
    });
  }
  return pool;
}

let schemaReady = false;
export async function ensureSchema() {
  const db = getPool();
  if (!db) throw new Error('DATABASE_URL not configured');
  if (schemaReady) return;
  await db.query(`
    CREATE TABLE IF NOT EXISTS simulations (
      id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      status      text NOT NULL DEFAULT 'created',
      scenario    jsonb NOT NULL,
      validation  jsonb,
      result      jsonb,
      error       text,
      created_at  timestamptz NOT NULL DEFAULT now(),
      updated_at  timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS simulations_status_idx ON simulations (status);
    CREATE INDEX IF NOT EXISTS simulations_created_idx ON simulations (created_at DESC);
  `);
  schemaReady = true;
}

export async function createSimulationRow(scenario, validation) {
  const db = getPool();
  const { rows } = await db.query(
    `INSERT INTO simulations (status, scenario, validation)
     VALUES ('created', $1, $2)
     RETURNING id, status, created_at`,
    [scenario, validation]
  );
  return rows[0];
}

export async function getSimulationRow(id) {
  const db = getPool();
  const { rows } = await db.query(`SELECT * FROM simulations WHERE id = $1`, [id]);
  return rows[0] || null;
}

export async function listSimulationRows(limit = 50) {
  const db = getPool();
  const { rows } = await db.query(
    `SELECT id, status, error, created_at, updated_at,
            scenario->'objective' AS objective,
            scenario->'region'->'domain_center' AS domain_center
       FROM simulations ORDER BY created_at DESC LIMIT $1`,
    [Math.min(limit, 200)]
  );
  return rows;
}
