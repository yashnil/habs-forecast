import { describe, expect, it } from "vitest";
import { ADVERSARIAL_ENGINE, GOLDEN, grade, loaderFor, nowFor, requestFor, type EngineCase } from "@/pilot/ask/evals/harness";
import { dailyCap, SpendMeter } from "@/pilot/ask/limits";
import { askLive, createAnthropicClient } from "@/pilot/ask/live";

/**
 * The same golden + adversarial sets against LIVE mode (real Claude calls, real spend).
 * Skipped unless explicitly enabled — never part of CI:
 *
 *   ANTHROPIC_API_KEY=… NERITIC_ASK_DAILY_USD_CAP=2 npm run eval:ask:live
 *
 * ~70 questions; most need one Sonnet-class call (~3–4k input tokens), some a retry, a few a
 * Haiku-class routing call. The spend meter stops at the cap and the remaining cases then grade the
 * deterministic fallback. Results print as a table; the run fails if any case fails.
 */
const enabled = process.env.NERITIC_ASK_LIVE_EVALS === "1" && !!process.env.ANTHROPIC_API_KEY;

describe.skipIf(!enabled)("live-mode evals (paid; opt-in)", () => {
  const meter = new SpendMeter(dailyCap(process.env));
  const rows: { id: string; mode: string; failures: string[] }[] = [];
  const cases: EngineCase[] = [...GOLDEN, ...ADVERSARIAL_ENGINE];
  for (const c of cases)
    it(c.id, { timeout: 60_000 }, async () => {
      const client = await createAnthropicClient(process.env.ANTHROPIC_API_KEY!);
      const loader = loaderFor(c);
      const now = nowFor(c);
      const a = await askLive(requestFor(c), { now, mode: "live", loader }, { client, meter });
      const facts = a.place ? ((await loader(a.place.place_id, now))?.today.facts ?? []) : [];
      const failures = grade(c, a, facts, now);
      rows.push({ id: c.id, mode: a.mode, failures });
      expect(failures, `${c.question}\n${a.lines.map((l) => l.text).join("\n")}`).toEqual([]);
    });
  it("summary", () => {
    const pass = rows.filter((r) => !r.failures.length).length;
    console.table(rows.map((r) => ({ id: r.id, mode: r.mode, result: r.failures.length ? r.failures.join("; ") : "pass" })));
    console.log(`live evals: ${pass}/${rows.length} passed; spent about $${meter.spentToday().toFixed(3)}`);
  });
});
