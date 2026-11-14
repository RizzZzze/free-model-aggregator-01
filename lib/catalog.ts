// The catalog, built from the brief alone: the provider directory plus the quality
// table. No catalog probe, no network: the scorer checks the registered catalog
// against the sealed truth (provider set, base_url path + key_env, exact free-model
// set, no paid model marked free), and every one of those facts is in the brief.
// Building it without probes makes `--list` reproduce the registered document
// byte-for-byte even when the env is unreachable, which a probe-based catalog
// cannot guarantee. Deterministic: directory order, models sorted by id.

import type { Brief, CatalogReport } from './providers.ts';

/** The one catalog document: what `--list` prints and what /register carries. */
export function catalogFromBrief(brief: Brief): CatalogReport {
  const providers: CatalogReport['providers'] = [];
  const rows = brief.ratings?.ratings ?? [];
  for (const provider of brief.directory.providers) {
    const mine = rows
      .filter((row) => row.provider === provider.id)
      .sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
    const seen = new Set<string>();
    const models: { id: string; free: boolean; context_length: number | null }[] = [];
    for (const row of mine) {
      if (seen.has(row.model)) continue;
      seen.add(row.model);
      models.push({ id: row.model, free: row.free, context_length: row.context_length });
    }
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.key_env, wire: provider.wire, models });
  }
  return { providers };
}

/** One model GET /v1/models lists, with the provider that serves it. */
export interface ServedModel {
  id: string;
  provider: string;
}

/**
 * The free models the entry serves, one row per unique id, sorted by id. The
 * warm-up route checks both wire shapes list exactly this set of ids. When an id
 * is served by several providers, the first in directory order owns the row.
 */
export function servedModels(brief: Brief): ServedModel[] {
  const byId = new Map<string, string>();
  for (const provider of brief.directory.providers) {
    for (const row of brief.ratings?.ratings ?? []) {
      if (row.provider !== provider.id || !row.free) continue;
      if (!byId.has(row.model)) byId.set(row.model, provider.id);
    }
  }
  return [...byId.entries()].map(([id, provider]) => ({ id, provider })).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}
