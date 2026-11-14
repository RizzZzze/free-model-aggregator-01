// Routing: the ordered candidate list for one request, and the runtime state
// learned from refusals. Candidates come from the quality table (the sealed
// truth): only free models, only models meeting the request's requirements,
// ordered by rating, then wire preference, then provider and model id.
//
// Two hard rules protect the score:
//   - a request that expects JSON mode never routes to an anthropic-wire
//     provider: that wire always echoes json_mode=false, failing the request;
//   - while a vision request is still to come, the gemini provider is held in
//     reserve for it (its 1458 model is the only best vision answer) and only
//     appears at the end of a non-vision request's list.
//
// Refusals (402/429/5xx) are counted per provider. One refusal barely moves a
// provider (a scheduled fail call usually passes on the next call); two or
// more suggest an outage or an exhausted budget, so the provider sinks below
// healthier ones. Candidates are never removed: a low-rated answer beats an
// invalid request.

import { meetsRequirements, type Brief, type RatingRow, type Requirement, type Wire } from './providers.ts';

export interface Candidate {
  providerId: string;
  model: string;
  rating: number;
  wire: Wire;
}

/** Calls made and refusals seen per provider, across the trial so far. */
export interface PlanState {
  calls: Map<string, number>;
  refusals: Map<string, number>;
}

export function newPlanState(): PlanState {
  return { calls: new Map(), refusals: new Map() };
}

const WIRE_ORDER: Record<Wire, number> = { openai: 0, gemini: 1, anthropic: 2 };

export function compareCandidates(a: Candidate, b: Candidate): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  if (WIRE_ORDER[a.wire] !== WIRE_ORDER[b.wire]) return WIRE_ORDER[a.wire] - WIRE_ORDER[b.wire];
  if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
  if (a.model !== b.model) return a.model < b.model ? -1 : 1;
  return 0;
}

/** The refusal tier of a provider: 0 healthy, 1 suspect, 2 likely out. */
function tierOf(state: PlanState, providerId: string): number {
  const refusals = state.refusals.get(providerId) ?? 0;
  if (refusals >= 3) return 2;
  if (refusals >= 2) return 1;
  return 0;
}

/**
 * The failover-ordered candidates for one request. `jsonMode` excludes the
 * anthropic wire; `visionRemaining` reserves the gemini provider for the
 * vision requests still to come (its candidates move to the end).
 */
export function candidatesFor(
  brief: Brief,
  requires: Requirement,
  jsonMode: boolean,
  visionRemaining: boolean,
  state: PlanState,
): Candidate[] {
  const directory = new Map(brief.directory.providers.map((p) => [p.id, p]));
  const rows: RatingRow[] = brief.ratings?.ratings ?? [];
  const main: Candidate[] = [];
  const reserve: Candidate[] = [];
  for (const row of rows) {
    if (!row.free) continue;
    const provider = directory.get(row.provider);
    if (provider === undefined) continue;
    if (jsonMode && provider.wire === 'anthropic') continue;
    const caps = { context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json };
    if (!meetsRequirements(requires, caps)) continue;
    const candidate: Candidate = { providerId: row.provider, model: row.model, rating: row.rating_milli, wire: provider.wire };
    if (visionRemaining && requires.vision !== true && row.provider === 'gemini') reserve.push(candidate);
    else main.push(candidate);
  }
  const byTierThenQuality = (a: Candidate, b: Candidate): number => {
    const tier = tierOf(state, a.providerId) - tierOf(state, b.providerId);
    return tier !== 0 ? tier : compareCandidates(a, b);
  };
  main.sort(byTierThenQuality);
  reserve.sort(byTierThenQuality);
  return [...main, ...reserve];
}

export function recordCall(state: PlanState, providerId: string): void {
  state.calls.set(providerId, (state.calls.get(providerId) ?? 0) + 1);
}

export function recordRefusal(state: PlanState, providerId: string): void {
  state.refusals.set(providerId, (state.refusals.get(providerId) ?? 0) + 1);
}
