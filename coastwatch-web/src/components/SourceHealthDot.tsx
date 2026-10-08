"use client";

import type { Manifest } from "@/generated/schema";
import type { FreshnessState } from "@/lib/freshness";
import { worstState } from "@/lib/health";
import { useNow } from "@/lib/useNow";

const COLOR: Record<FreshnessState, string> = {
  current: "var(--cw-good)",
  stale: "var(--cw-warning)",
  historical: "var(--cw-serious)",
  unavailable: "var(--cw-neutral)",
};
const TEXT: Record<FreshnessState, string> = {
  current: "all data sources current",
  stale: "some data is stale",
  historical: "some data is out of date",
  unavailable: "some data is unavailable",
};

export function SourceHealthDot({ manifest }: { manifest: Manifest | null }) {
  const now = useNow();
  const state = now ? worstState(manifest, now) : null;
  return (
    <span className="inline-flex items-center" data-testid="source-health" data-state={state ?? "checking"}>
      <span className="h-2 w-2 rounded-full" style={{ background: state ? COLOR[state] : "var(--cw-ink-3)" }} aria-hidden />
      <span className="sr-only">{state ? TEXT[state] : "checking data freshness"}</span>
    </span>
  );
}
