import type { PortFact } from "@/pilot/contracts";
import { diffFacts, fisheryLabel, type FactChange, type FactSnap } from "@/pilot/facts";
import { day, pacificYmd, time } from "@/nx/ds/format";

/**
 * "Since your last visit", computed (journeys.md Journey 2): the facts this device saw at its last visit
 * to the port (src/pilot/today/prefs.ts `readSeen`) against the facts now, by PortFact.id
 * (src/pilot/facts.ts `diffFacts`). Status, notices, whale measures, tests, model runs and NWS
 * advisories count; the hourly sea, tide and buoy do not. No earlier visit: no line (never invented).
 */
export type ChangeLine = { since: string; text: string; changes: Array<FactChange & { words: string }> };

const fisheryOf = (id: string) => id.replace(/^status:/, "") as Parameters<typeof fisheryLabel>[0];
const strip = (s: string) => s.trim().replace(/[.]+$/, "");

/** One change in a few words: "Dungeness crab (commercial) now delayed", "new CDPH test, Monterey Bay". */
export function changeWords(c: FactChange): string {
  const f = (c.after ?? c.before) as FactSnap;
  if (f.kind === "status") {
    const label = fisheryLabel(fisheryOf(f.id));
    return c.change === "gone" ? `${label}: status no longer listed` : `${label} is now ${(c.after?.value ?? "changed").toLowerCase()}`;
  }
  if (f.kind === "test") {
    const site = strip(f.headline.split(/ (tested|:)/)[0] ?? "");
    return c.change === "gone" ? `${site} test no longer listed` : `new CDPH test, ${site}`;
  }
  if (f.kind === "notice") return c.change === "gone" ? "a notice is no longer listed" : `${c.change === "new" ? "new" : "changed"} notice: ${strip(f.headline)}`;
  if (f.kind === "whale") return "whale measures changed";
  if (f.kind === "bloom_forecast") return "a newer C-HARM run";
  if (f.kind === "bloom_outlook") return "a newer algae outlook";
  if (f.id.startsWith("weather:alert:")) return c.change === "gone" ? "an NWS advisory has ended" : strip(f.headline);
  return strip(f.headline);
}

/** When: "Oct 8", or "7:02 AM PT" for a visit earlier the same day. */
export function sinceWords(since: string, now: string): string {
  return pacificYmd(since) === pacificYmd(now) ? time(since) : day(since, { ref: now });
}

export function changeLine(prev: { at: string; facts: FactSnap[] } | null, facts: PortFact[], now: string): ChangeLine | null {
  if (!prev) return null;
  const changes = diffFacts(prev.facts, facts).map((c) => ({ ...c, words: changeWords(c) }));
  const when = sinceWords(prev.at, now);
  if (!changes.length) return { since: prev.at, text: `Since ${when}: no change in season status, tests, notices or models.`, changes };
  const first = changes[0].words;
  const text = `Since ${when}: ${first}${changes.length > 1 ? `, and ${changes.length - 1} more` : ""}.`;
  return { since: prev.at, text, changes };
}
