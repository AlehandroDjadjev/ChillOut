/**
 * ChillOut API — scenario validation + WRF simulation job queue.
 *
 * Action-based dispatch (POST JSON { action, ... }). The Lambda NEVER runs WRF (it can't —
 * 30s/512MB/no MPI); it validates scenarios and queues jobs in Postgres. A separate WRF
 * worker daemon on the VM picks up `status='created'` rows and runs the real model.
 *
 * Env: DATABASE_URL (Postgres), STORAGE_BUCKET (S3), OPENKBS_PROJECT_ID.
 */
import { validateScenario, STATUSES } from './lib/schema.mjs';
import { ensureSchema, createSimulationRow, getSimulationRow, listSimulationRows } from './lib/db.mjs';

const headers = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
};
const json = (statusCode, data) => ({ statusCode, headers, body: JSON.stringify(data) });

export const handler = async (event) => {
  if (event.requestContext?.http?.method === 'OPTIONS') {
    return { statusCode: 204, headers, body: '' };
  }

  let body;
  try {
    body = JSON.parse(event.body || '{}');
  } catch {
    return json(400, { error: 'Invalid JSON body' });
  }
  const { action } = body;

  try {
    switch (action) {
      case 'hello':
        return json(200, { message: 'ChillOut API', timestamp: new Date().toISOString() });

      case 'status':
        return json(200, {
          postgres: !!process.env.DATABASE_URL,
          storage: !!process.env.STORAGE_BUCKET,
          projectId: process.env.OPENKBS_PROJECT_ID,
        });

      case 'migrate':
        await ensureSchema();
        return json(200, { migrated: true });

      case 'createSimulation': {
        const scenario = body.scenario;
        const { valid, errors } = validateScenario(scenario);
        if (!valid) return json(400, { valid: false, errors });
        await ensureSchema();
        const row = await createSimulationRow(scenario, { valid: true, errors: [] });
        return json(201, { id: row.id, status: row.status, created_at: row.created_at });
      }

      case 'validateScenario': {
        // Dry-run validation (no row created) — used by the editor for live feedback.
        const { valid, errors } = validateScenario(body.scenario);
        return json(200, { valid, errors });
      }

      case 'getSimulation':
      case 'getStatus':
      case 'getResults': {
        if (!body.id) return json(400, { error: 'id required' });
        await ensureSchema();
        const row = await getSimulationRow(body.id);
        if (!row) return json(404, { error: 'simulation not found' });
        if (action === 'getStatus') {
          return json(200, { id: row.id, status: row.status, error: row.error, updated_at: row.updated_at });
        }
        if (action === 'getResults') {
          return json(200, { id: row.id, status: row.status, result: row.result, error: row.error });
        }
        return json(200, { simulation: row });
      }

      case 'listSimulations': {
        await ensureSchema();
        const rows = await listSimulationRows(body.limit);
        return json(200, { simulations: rows });
      }

      default:
        return json(400, {
          error: 'Unknown action',
          available: ['hello', 'status', 'migrate', 'createSimulation', 'validateScenario',
            'getSimulation', 'getStatus', 'getResults', 'listSimulations'],
          statuses: STATUSES,
        });
    }
  } catch (error) {
    return json(500, { error: error.message });
  }
};
