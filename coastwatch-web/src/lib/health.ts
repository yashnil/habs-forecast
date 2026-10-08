import type { Manifest } from "@/generated/schema";
import { classifySource, type FreshnessState } from "@/lib/freshness";

const RANK: Record<FreshnessState, number> = { current: 0, stale: 1, historical: 2, unavailable: 3 };

/** Worst freshness across sources; a failed last update attempt counts as at least stale. */
export function worstState(m: Manifest | null, now: Date): FreshnessState {
  if (!m || m.sources.length === 0) return "unavailable";
  let worst: FreshnessState = "current";
  for (const s of m.sources) {
    let st = classifySource(s, now).state;
    if (s.outcome === "failed" && RANK[st] < RANK.stale) st = "stale";
    if (RANK[st] > RANK[worst]) worst = st;
  }
  return worst;
}
