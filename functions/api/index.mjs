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
import { fetchSentinel2Png } from './lib/sentinel2.mjs';
import { ensureInferenceSchema, createInferenceJob, getInferenceJob } from './lib/inference_db.mjs';
import { buildSample } from './lib/build_sample.mjs';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const headers = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
};
const json = (statusCode, data) => ({ statusCode, headers, body: JSON.stringify(data) });

const SAFE_NAME = /^[A-Za-z0-9._-]+$/;

// Presigned S3 PUT URL (AWS SDK, the documented OpenKBS pattern). Objects land under the CDN
// `media/` prefix so the worker can pull them later with `openkbs storage download`.
const s3 = new S3Client({});
async function getUploadUrl({ jobUuid, filename, contentType }) {
  const bucket = process.env.STORAGE_BUCKET;
  if (!bucket) throw new Error('Storage not configured on this function.');
  if (!jobUuid || !SAFE_NAME.test(jobUuid)) throw new Error('Invalid jobUuid.');
  if (!filename || !SAFE_NAME.test(filename)) throw new Error('Invalid filename.');
  const key = `media/inference/${jobUuid}/${filename}`;
  const command = new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    ContentType: contentType || 'application/octet-stream',
  });
  const uploadUrl = await getSignedUrl(s3, command, { expiresIn: 3600 });
  return { uploadUrl, key };
}

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

      case 'sentinel2': {
        const image_data_url = await fetchSentinel2Png({
          bbox: body.bbox,
          date: body.date,
          mode: body.mode,
          width: body.width,
          height: body.height,
        });
        return json(200, { image_data_url, mode: body.mode || 'cloud_mask', date: body.date });
      }

      case 'buildSample': {
        const sample = await buildSample({
          place: body.place,
          lat: body.lat,
          lon: body.lon,
          date: body.date,
          bbox: body.bbox,
          windowDays: body.windowDays,
        });
        return json(200, { sample });
      }

      case 'getUploadUrl': {
        const { uploadUrl, key } = await getUploadUrl({
          jobUuid: body.jobUuid,
          filename: body.filename,
          contentType: body.contentType,
        });
        return json(200, { uploadUrl, key });
      }

      case 'createInference': {
        if (!body.checkpoint_key) return json(400, { error: 'checkpoint_key required' });
        if (!body.mask_data_url) return json(400, { error: 'mask_data_url required' });
        if (!Array.isArray(body.raw_features) || body.raw_features.length !== 14) {
          return json(400, { error: 'raw_features must be a 14-number array' });
        }
        await ensureInferenceSchema();
        const row = await createInferenceJob({
          checkpoint_key: body.checkpoint_key,
          stats_key: body.stats_key,
          mask_data_url: body.mask_data_url,
          raw_features: body.raw_features,
          target_temp: body.target_temperature_c,
          place: body.place,
          date: body.date,
        });
        return json(201, { id: row.id, status: row.status, created_at: row.created_at });
      }

      case 'getInference': {
        if (!body.id) return json(400, { error: 'id required' });
        await ensureInferenceSchema();
        const row = await getInferenceJob(body.id);
        if (!row) return json(404, { error: 'inference job not found' });
        return json(200, { id: row.id, status: row.status, result: row.result, error: row.error });
      }

      default:
        return json(400, {
          error: 'Unknown action',
          available: ['hello', 'status', 'migrate', 'createSimulation', 'validateScenario',
            'getSimulation', 'getStatus', 'getResults', 'listSimulations', 'sentinel2',
            'buildSample', 'getUploadUrl', 'createInference', 'getInference'],
          statuses: STATUSES,
        });
    }
  } catch (error) {
    return json(500, { error: error.message });
  }
};
