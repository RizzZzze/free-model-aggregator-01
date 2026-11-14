// Client responses: JSON and synthesized SSE for both client wires. The entry
// always calls providers with JSON (nothing scores whether an upstream call was
// streamed), so a streaming client answer is synthesized here from the complete
// text. The chunking is a pure function of the text (fixed 13-character pieces),
// which keeps a replay byte-identical. Every response carries the provider's
// usage untouched and a family-0 stop reason (the providers always answer
// family 0, and the client always expects it).

import type { Json } from './providers.ts';
import type { ServedModel } from './catalog.ts';

/** Fixed timestamps: nothing in a reply reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** Characters per synthesized stream piece (the providers' own piece size). */
const CHUNK_CHARS = 13;

function chunksOf(text: string): string[] {
  const points = Array.from(text);
  const out: string[] = [];
  for (let i = 0; i < points.length; i += CHUNK_CHARS) out.push(points.slice(i, i + CHUNK_CHARS).join(''));
  return out;
}

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** A Chat Completions JSON answer. */
export function openAIJson(model: string, text: string, inputTokens: number, outputTokens: number): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  };
}

/** A Messages JSON answer. */
export function anthropicJson(model: string, text: string, inputTokens: number, outputTokens: number): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
  };
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream: content chunks, exactly one finish chunk
 * carrying the usage, then [DONE] exactly once and last. Nothing after finish.
 */
export function openAISSE(model: string, text: string, inputTokens: number, outputTokens: number): string {
  let out = '';
  const chunks = chunksOf(text);
  for (let i = 0; i < chunks.length; i++) {
    out += sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta: i === 0 ? { role: 'assistant', content: chunks[i] } : { content: chunks[i] }, finish_reason: null }],
    });
  }
  out += sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/**
 * A Messages SSE stream in the canonical order: message_start, one text
 * content block (start, deltas, stop), message_delta with the stop reason and
 * output usage, message_stop. Every frame carries an event name.
 */
export function anthropicSSE(model: string, text: string, inputTokens: number, outputTokens: number): string {
  let out = '';
  out += sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregate',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    },
  });
  out += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const chunk of chunksOf(text)) {
    out += sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: chunk } });
  }
  out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: outputTokens },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}
