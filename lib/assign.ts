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
// 1: this module assigns each request its first-choice provider up front by
// minimizing expected refusals plus a small quality penalty (see failProb and
// planAssignment below). The greedy order still serves as the per-request
// failover list (plan.ts). Quality 98 still totals 99, so the plan may trade
// a little rating for a lot fewer calls; the cost function spends the trade
// where it costs least by assigning the most constrained requests first.
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

/** Published per-provider budget ranges from the kit's env code (server.mjs PROVIDERS table). */
const BUDGET_RANGE: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

/**
 * Approximate probability that a provider's d-th counted call refuses, from
 * the draw mechanics in the kit's generator (draw.mjs): budget uniform over
 * the range, 1-2 fail numbers uniform in [2, budget-1], one provider forced
 * to fail at call 2, one provider with an outage starting uniform in
 * [2, budget-1]. Call 1 never refuses. The approximation only ranks slots;
 * exact probabilities are unknowable without the sealed seed.
 */
function failProb(providerId: string, depth: number): number {
  if (depth <= 1) return 0;
  const [lo, hi] = BUDGET_RANGE[providerId] ?? [4, 16];
  const meanBudget = (lo + hi) / 2;
  const pOver = depth > hi ? 1 : depth <= lo ? 0 : (depth - lo) / (hi - lo + 1);
  const pFail = depth <= meanBudget - 1 ? 1.5 / (meanBudget - 2) : 0;
  const pDesignated = depth === 2 ? 0.1 * (1 - pFail) : 0;
  const pOutage = 0.1 * Math.min(1, Math.max(0, (depth - 1) / (meanBudget - 2)));
  return 1 - (1 - pOver) * (1 - pFail) * (1 - pDesignated) * (1 - pOutage);
}

/** The quality penalty per weighted rating point in the assignment cost. Picked on 20 eval instances and confirmed on 20 fresh holdout seeds (mean 22.90 calls on both sets, score 99 on all 40); 0 breaches quality, 0.001+ costs ~0.7 more calls. */
const LAMBDA = 0.0005;

/**
 * Cost-based assignment: requests are assigned most constrained first
 * (fewest eligible models, then heavier weight, then earliest); each takes
 * the candidate minimizing expected refusals plus LAMBDA times the weighted
 * quality loss (best rating minus the candidate's, times the request weight).
 * Depth-1 slots cost nothing, depth-2 on a small-budget provider costs more
 * than depth-3 on a big one, so the plan spends safe slots first and trades
 * rating for safety only where the cost says it is worth it.
 */
export function planAssignment(brief: Brief, rows: PlannedRequest[]): Map<number, Candidate> {
  const perRow: Candidate[][] = rows.map((r) => eligibleCandidates(brief, r.requires));
  const order = rows.map((_, i) => i).sort((a, b) => {
    if (perRow[a].length !== perRow[b].length) return perRow[a].length - perRow[b].length;
    if (rows[a].weight !== rows[b].weight) return rows[b].weight - rows[a].weight;
    return a - b;
  });
  const depth = new Map<string, number>();
  const assignment = new Map<number, Candidate>();
  for (const i of order) {
    const cands = perRow[i];
    if (cands.length === 0) continue;
    const best = Math.max(...cands.map((c) => c.rating));
    let chosen: Candidate | undefined;
    let chosenCost = Infinity;
    for (const c of cands) {
      const d = (depth.get(c.providerId) ?? 0) + 1;
      const cost = failProb(c.providerId, d) + LAMBDA * rows[i].weight * (best - c.rating);
      if (cost < chosenCost - 1e-12 || (Math.abs(cost - chosenCost) <= 1e-12 && chosen !== undefined && compareCandidates(c, chosen) < 0)) {
        chosen = c;
        chosenCost = cost;
      }
    }
    if (chosen === undefined) continue;
    assignment.set(i, chosen);
    depth.set(chosen.providerId, (depth.get(chosen.providerId) ?? 0) + 1);
  }
  return assignment;
}
