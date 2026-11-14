// One inference call to one provider, in the provider's own wire, always asking
// for JSON. A 2xx with a JSON body is a success. A 402, 429 or 5xx is a refusal:
// the request fails over to the next candidate. Anything else (a 400/401, a
// body that is not JSON, a network error, a timeout) is an error: the request
// still fails over, but the provider is not counted as refusing, because the
// fault is ours or the network's, not its budget.

import { authHeaders, isRecord, joinBase, type Json, type ProviderDirectoryEntry } from './providers.ts';
import type { UpstreamWire } from './translate.ts';

export type CallOutcome =
  | { kind: 'ok'; json: Json }
  | { kind: 'refused'; status: number }
  | { kind: 'error'; message: string };

export async function callUpstream(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  model: string,
  wire: UpstreamWire,
  body: Json,
  anthropicVersion: string,
  timeoutMs: number,
): Promise<CallOutcome> {
  const base = joinBase(envUrl, provider.base);
  const url =
    wire === 'gemini'
      ? `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`
      : wire === 'anthropic'
        ? `${base}/v1/messages`
        : `${base}/v1/chat/completions`;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (wire === 'anthropic') headers['anthropic-version'] = anthropicVersion;
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(timeoutMs) });
  } catch (err) {
    return { kind: 'refused', status: 0 };
  }
  if (res.status === 200) {
    try {
      const value: unknown = await res.json();
      if (isRecord(value)) return { kind: 'ok', json: value };
      return { kind: 'error', message: `HTTP 200 with a non-object body from ${provider.id}` };
    } catch (err) {
      return { kind: 'error', message: `HTTP 200 with an unreadable body from ${provider.id}: ${(err as Error).message}` };
    }
  }
  await res.text().catch(() => undefined);
  if (res.status === 402 || res.status === 429 || res.status >= 500) return { kind: 'refused', status: res.status };
  return { kind: 'error', message: `${provider.id} answered HTTP ${res.status}` };
}
