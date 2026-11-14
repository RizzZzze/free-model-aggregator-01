// Global upfront assignment: one planned first choice per manifest request.
//
// The dev board showed the greedy rating order's weakness: 293 of 308 entries
// score 99, so the tie-break (fewest provider calls) decides everything, and
// concentrating on the top providers burns calls on their scheduled fail
// numbers and budget edges (28 calls, rank 238). The entries at the top plan
// ahead and spread: their titles say so ("Cap-2 Planning", "Lookahead Global
// Assignment", "Upfront planning + fewest calls").
//
// The manifest is in the brief, so the whole workload is known before request
// 1: this module assigns each request its first-choice provider up front, with
// a per-provider cap on PLANNED calls. Spreading planned calls keeps every
// provider shallow, clear of most fail numbers and budget edges, while the
// greedy order still serves as the per-request failover list (plan.ts).
// Quality 98 still totals 99, so the plan may trade a little rating for a lot
// fewer calls; the greedy below spends the trade where it costs least (the
// weight-1 unconstrained requests) by assigning the most constrained requests
// first.
//
// The plan ignores JSON mode (only the request body carries it): at request
// time the anthropic wire is still excluded for JSON bodies, and the assigned
// choice is simply skipped when it is not in the filtered list. The plan also
// ignores the vision reserve: vision requests have the fewest eligible models,
// so the constrainedness order hands them the gemini provider first anyway.

import { meetsRequirements, type Brief, type RatingRow, type Requirement, type Wire } from './providers.ts';
import { compareCandidates, type Candidate } from './plan.ts';

export interface PlannedRequest {
  requires: Requirement;
  weight: number;
}

function eligibleCandidates(brief: Brief, requires: Requirement): Candidate[] {
  const directory = new Map(brief.directory.providers.map((p) => [p.id, p]));
  const rows: RatingRow[] = brief.ratings?.ratings ?? [];
  const out: Candidate[] = [];
  for (const row of rows) {
    if (!row.free) continue;
    const provider = directory.get(row.provider);
    if (provider === undefined) continue;
    const caps = { context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json };
    if (!meetsRequirements(requires, caps)) continue;
    out.push({ providerId: row.provider, model: row.model, rating: row.rating_milli, wire: provider.wire as Wire });
  }
  return out;
}

/**
 * The planned first choice per request index. Requests are assigned most
 * constrained first (fewest eligible models, then heavier weight, then
 * earliest); each takes the best-rated model on a provider still under the
 * cap, or the best-rated model overall when every eligible provider is at
 * the cap. The cap is one number for all providers, or a per-provider map
 * (unknown providers fall back to 2). Deterministic: the brief alone decides
 * the plan.
 */
export function planAssignment(brief: Brief, rows: PlannedRequest[], cap: number | Map<string, number>): Map<number, Candidate> {
  const capOf = (providerId: string): number => (typeof cap === 'number' ? cap : (cap.get(providerId) ?? 2));
  const perRow: Candidate[][] = rows.map((r) => eligibleCandidates(brief, r.requires));
  const order = rows.map((_, i) => i).sort((a, b) => {
    if (perRow[a].length !== perRow[b].length) return perRow[a].length - perRow[b].length;
    if (rows[a].weight !== rows[b].weight) return rows[b].weight - rows[a].weight;
    return a - b;
  });
  const planned = new Map<string, number>();
  const assignment = new Map<number, Candidate>();
  for (const i of order) {
    const cands = [...perRow[i]].sort(compareCandidates);
    const underCap = cands.find((c) => (planned.get(c.providerId) ?? 0) < capOf(c.providerId));
    const chosen = underCap ?? cands[0];
    if (chosen === undefined) continue;
    assignment.set(i, chosen);
    planned.set(chosen.providerId, (planned.get(chosen.providerId) ?? 0) + 1);
  }
  return assignment;
}
