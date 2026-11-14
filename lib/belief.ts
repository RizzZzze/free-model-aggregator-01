// Bayesian belief over each provider's hidden draw: the budget, the fail
// numbers, and the outage start. The draw mechanics are in the kit's
// generator (draw.mjs): budget uniform over the provider's published range,
// 1-2 fail numbers uniform in [2, budget-1], one provider forced to fail at
// call 2, one provider with an outage starting uniform in [2, budget-1].
//
// The observation model is deterministic given the hidden state: a counted
// call at depth d refuses exactly when d is past the budget, at or past the
// outage start, or one of the fail numbers. So the posterior is just the set
// of possible worlds consistent with every outcome so far, weighted by the
// draw prior — no floating-point inference machinery, only set filtering in
// a fixed enumeration order, which keeps replays byte-identical.
//
// The designated-provider and outage-provider constraints (exactly one of
// each per trial) are approximated as independent 0.1 priors per provider:
// the joint constraint would sharpen inference slightly, at much higher
// complexity. The designated mechanism needs no state of its own: it only
// raises the prior that 2 is a fail number, which the fail-set weights below
// already encode.

import { BUDGET_RANGE } from './assign.ts';

interface World {
  budget: number;
  fails: number[];
  outage: number | null;
  weight: number;
}

export interface Belief {
  worlds: Map<string, World[]>;
  depths: Map<string, number>;
}

function enumerateWorlds(lo: number, hi: number): World[] {
  const worlds: World[] = [];
  for (let budget = lo; budget <= hi; budget++) {
    const positions: number[] = [];
    for (let d = 2; d <= budget - 1; d++) positions.push(d);
    const sets: number[][] = [];
    for (const a of positions) sets.push([a]);
    for (let i = 0; i < positions.length; i++) {
      for (let j = i + 1; j < positions.length; j++) sets.push([positions[i], positions[j]]);
    }
    const n1 = positions.length;
    const n2 = (n1 * (n1 - 1)) / 2;
    for (const fails of sets) {
      const sizeWeight = fails.length === 1 ? 0.5 / n1 : 0.5 / n2;
      const condCount = fails.length === 1 ? 1 : n1 - 1;
      const condWeight = fails.includes(2) ? 0.5 / condCount : 0;
      const failWeight = 0.9 * sizeWeight + 0.1 * condWeight;
      const outageOptions: (number | null)[] = [null, ...positions];
      for (const outage of outageOptions) {
        const outageWeight = outage === null ? 0.9 : 0.1 / positions.length;
        worlds.push({ budget, fails, outage, weight: (1 / (hi - lo + 1)) * failWeight * outageWeight });
      }
    }
  }
  return worlds;
}

export function newBelief(): Belief {
  const worlds = new Map<string, World[]>();
  for (const [id, [lo, hi]] of Object.entries(BUDGET_RANGE)) worlds.set(id, enumerateWorlds(lo, hi));
  return { worlds, depths: new Map() };
}

function refuses(world: World, depth: number): boolean {
  return depth > world.budget || (world.outage !== null && depth >= world.outage) || world.fails.includes(depth);
}

/**
 * Record one counted call's outcome, in call order. Only 'ok' and 'refused'
 * outcomes are counted calls; transport errors are never observed (their
 * counting is unknown). Depth is the provider's next counted call number.
 */
export function observe(belief: Belief, providerId: string, wasRefused: boolean): void {
  const worlds = belief.worlds.get(providerId);
  if (worlds === undefined) return;
  const depth = (belief.depths.get(providerId) ?? 0) + 1;
  const kept = worlds.filter((w) => refuses(w, depth) === wasRefused);
  belief.worlds.set(providerId, kept.length > 0 ? kept : worlds);
  belief.depths.set(providerId, depth);
}

/** Posterior probability that the provider's next counted call at `depth` refuses. */
export function refusalProb(belief: Belief, providerId: string, depth: number): number {
  const worlds = belief.worlds.get(providerId);
  if (worlds === undefined || worlds.length === 0) return 0.5;
  let total = 0;
  let refusing = 0;
  for (const w of worlds) {
    total += w.weight;
    if (refuses(w, depth)) refusing += w.weight;
  }
  return total > 0 ? refusing / total : 0.5;
}

/** Counted calls observed so far per provider: the depth the next call would take. */
export function currentDepths(belief: Belief): Map<string, number> {
  return new Map(belief.depths);
}
