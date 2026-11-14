// One server, two client wires, ten simulated free-tier providers.
//
// The trial gives this server a brief (the provider directory, the quality
// table, the workload manifest) and an env URL. It reads the brief, builds one
// catalog from it, listens on a free port, registers with the env, serves the
// client's twenty requests one at a time, polls GET /work, and exits 0 the
// moment the env says done. A process still running when the wall ends scores
// zero, so nothing here waits on anything but the poll.
//
// Routing: for each request, the candidates are the free models meeting its
// requirements, ordered by the quality table's rating (plan.ts). Each candidate
// is tried in order until one answers; a refusal (402/429/5xx) fails over to
// the next. Providers are always called with JSON, whatever the client asked
// for: nothing scores whether an upstream call was streamed, and a streaming
// client answer is synthesized from the complete text (render.ts), which keeps
// a replay byte-identical. Request fields pass through verbatim across wires
// (translate.ts): the providers echo what they were sent, and the scorer
// compares the echo with the client's original body.
//
// `node main.ts --list` prints the same catalog document the register body
// carries, byte for byte, and exits 0 before any server starts. Outside a
// trial it prints {"providers":[]}.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the
//                      register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the
//                      capability matcher (from the kit's starter, kept).
//   lib/catalog.ts     the catalog and the served-model list, from the brief.
//   lib/plan.ts        the candidate order and the refusal-learned state.
//   lib/translate.ts   request translation across wires, answer extraction.
//   lib/render.ts      client JSON and synthesized SSE, both wires.
//   lib/upstream.ts    one provider call, classified.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { catalogFromBrief, servedModels } from './lib/catalog.ts';
import { candidatesFor, newPlanState, recordCall, recordRefusal, type PlanState } from './lib/plan.ts';
import {
  extractAnswer,
  fromAnthropic,
  fromOpenAI,
  toAnthropic,
  toGemini,
  toOpenAI,
  type ClientWire,
} from './lib/translate.ts';
import {
  anthropicError,
  anthropicJson,
  anthropicModelList,
  anthropicSSE,
  openAIError,
  openAIJson,
  openAIModelList,
  openAISSE,
} from './lib/render.ts';
import { callUpstream } from './lib/upstream.ts';
import { isRecord, readBrief, type Brief, type CatalogReport, type Json, type Requirement } from './lib/providers.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The anthropic-version sent upstream when the client sent none. */
const DEFAULT_ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = brief.error === null ? catalogFromBrief(brief) : { providers: [] };
const planState: PlanState = newPlanState();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): { requires: Requirement; index: number } {
  const manifest = brief.workload?.manifest ?? [];
  const index = manifestIndex;
  manifestIndex += 1;
  const entry = manifest[index];
  if (entry !== undefined) return { requires: entry.requires, index };
  return { requires: requirementsFromBody(body), index };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(brief);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One client inference request: translate it, try the ordered candidates until
 * one answers, and render the answer in the client's wire (JSON or synthesized
 * SSE, as asked). A refusal fails over to the next candidate; if every
 * candidate refuses, the client gets a 502 and the trial goes on.
 */
async function serveRequest(wire: ClientWire, body: Json, anthropicVersion: string, res: ServerResponse): Promise<void> {
  const canon = wire === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  const { requires, index } = requirementsFor(body);
  const manifest = brief.workload?.manifest ?? [];
  const visionRemaining = manifest.slice(index + 1).some((r) => r.requires.vision === true);
  const candidates = candidatesFor(brief, requires, canon.jsonMode !== null, visionRemaining, planState);
  const directory = new Map(brief.directory.providers.map((p) => [p.id, p]));
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  for (const candidate of candidates) {
    const provider = directory.get(candidate.providerId);
    if (provider === undefined) continue;
    const upstreamBody =
      candidate.wire === 'openai' ? toOpenAI(canon, candidate.model) : candidate.wire === 'anthropic' ? toAnthropic(canon, candidate.model) : toGemini(canon);
    recordCall(planState, candidate.providerId);
    const outcome = await callUpstream(ENV_URL, provider, candidate.model, candidate.wire, upstreamBody, anthropicVersion, UPSTREAM_TIMEOUT_MS);
    if (outcome.kind === 'ok') {
      const answer = extractAnswer(candidate.wire, outcome.json);
      if (canon.stream) {
        const text =
          wire === 'openai'
            ? openAISSE(requested, answer.text, answer.inputTokens, answer.outputTokens)
            : anthropicSSE(requested, answer.text, answer.inputTokens, answer.outputTokens);
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(text);
      } else {
        sendJson(
          res,
          200,
          wire === 'openai'
            ? openAIJson(requested, answer.text, answer.inputTokens, answer.outputTokens)
            : anthropicJson(requested, answer.text, answer.inputTokens, answer.outputTokens),
        );
      }
      return;
    }
    if (outcome.kind === 'refused') {
      recordRefusal(planState, candidate.providerId);
      log(`${candidate.providerId} refused (HTTP ${outcome.status}); failing over`);
    } else {
      log(`upstream error: ${outcome.message}; failing over`);
    }
  }
  log(`no candidate served request ${index + 1}`);
  sendError(res, wire, 502, wire === 'openai' ? 'upstream_unavailable' : 'api_error', 'every eligible provider refused this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const wire: ClientWire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) {
      sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
      return;
    }
    const version = req.headers['anthropic-version'];
    await serveRequest(wire, body, typeof version === 'string' ? version : DEFAULT_ANTHROPIC_VERSION, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels(brief).length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
