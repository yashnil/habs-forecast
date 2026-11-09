import { PILOT_PORTS, type PortFact, type PortToday } from "@/pilot/contracts";

/**
 * STUB — sample PortToday facts, typed against contracts.ts, used ONLY until the Today track's
 * src/pilot/facts.ts lands (branch pilot/today). Every fact here names its source as a pilot sample,
 * and pages that use it say "sample facts" (source.ts → `stub: true`). It is never presented as
 * live data. The wording follows the plan's 2026-11-08 reading of CDFW/CDPH, but the integrator
 * must replace this with the real facts builder (see source.ts).
 */
export const STUB_SOURCE = "Pilot sample (not live data)";
export const STUB_GENERATED_AT = "2026-11-08T14:00:00Z";

function factsFor(place_id: string): PortFact[] {
  const src = (name: string, url: string | null) => ({ name: `${STUB_SOURCE} · ${name}`, url });
  const monterey = place_id !== "port-452";
  const facts: PortFact[] = [
    {
      id: `${place_id}:status:dungeness_commercial`,
      place_id,
      kind: "status",
      severity: "red",
      headline: "Commercial Dungeness: delayed to at least Jan 1 (whale entanglement)",
      detail: "Next CDFW risk assessment Dec 15–18.",
      basis: "official",
      verified: false,
      source: src("CDFW", "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries"),
      as_of: "2026-11-08",
    },
    {
      id: `${place_id}:test:dungeness`,
      place_id,
      kind: "test",
      severity: "neutral",
      headline: monterey ? "Monterey Bay Dungeness tested below the action level" : "Half Moon Bay Dungeness tested below the action level",
      value: monterey ? "avg 5.5 ppm (limit 30)" : "avg 3–4 ppm (limit 30)",
      basis: "measured",
      verified: null,
      source: src("CDPH", "https://www.cdph.ca.gov/Programs/CEH/DFDCS/Pages/FDBPrograms/FoodSafetyProgram/DomoicAcid.aspx"),
      as_of: "2026-09-28",
    },
    {
      id: `${place_id}:whale:ramp`,
      place_id,
      kind: "whale",
      severity: "red",
      headline: "RAMP: commercial season delay in Zones 1–5",
      detail: "Next assessment Dec 15–18.",
      basis: "official",
      verified: false,
      source: src("CDFW RAMP", "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries"),
      as_of: "2026-11-08",
    },
    {
      id: `${place_id}:bloom_forecast:charm`,
      place_id,
      kind: "bloom_forecast",
      severity: "neutral",
      headline: "C-HARM domoic acid probability low for the next 3 days",
      value: "10%",
      basis: "forecast",
      verified: null,
      source: src("NOAA C-HARM", "https://coastwatch.noaa.gov"),
      as_of: "2026-11-08",
      valid_for: { from: "2026-11-08", to: "2026-11-11" },
    },
    {
      id: `${place_id}:data_age`,
      place_id,
      kind: "data_age",
      severity: "neutral",
      headline: "Neritic's data for this port was refreshed today",
      basis: "derived",
      verified: null,
      source: src("Neritic", null),
      as_of: STUB_GENERATED_AT,
    },
  ];
  if (monterey)
    facts.splice(1, 0, {
      id: `${place_id}:status:anchovy_sardine`,
      place_id,
      kind: "status",
      severity: "amber",
      headline: "Anchovy and sardine: CDPH consumption restriction, Pigeon Point to Point Lobos",
      basis: "official",
      verified: false,
      source: src("CDPH", "https://www.cdph.ca.gov"),
      as_of: "2026-09-11",
    });
  return facts;
}

export function stubPortToday(place_id: string): PortToday | null {
  const p = PILOT_PORTS.find((x) => x.place_id === place_id);
  if (!p) return null;
  return { place_id, label: p.label, generated_at: STUB_GENERATED_AT, data_generated_at: STUB_GENERATED_AT, facts: factsFor(place_id) };
}
