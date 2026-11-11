import type { PortFact } from "@/pilot/contracts";
import type { Kind, Provenance } from "@/nx/ds";

/**
 * Every value on Today carries a Provenance (README principle 5, T1): kind, source, date, verification.
 * Kind words are T1's own: an official record is "official" (and the stamp adds "not verified", T38), a
 * buoy or a CDPH lab result is "measured", NWS and CO-OPS are "prediction", C-HARM is "model", the
 * chlorophyll outlook is "outlook".
 */
export function kindOfFact(f: Pick<PortFact, "basis" | "kind">): Kind {
  switch (f.basis) {
    case "official":
      return "official";
    case "measured":
      return "measured";
    case "outlook":
      return "outlook";
    case "analysis":
      return "analysis";
    case "forecast":
      return f.kind === "bloom_forecast" ? "model" : "prediction";
    default:
      return "estimated";
  }
}

/** The date role a stamp names for a fact: a record was checked, a test sampled, a forecast issued. */
function roleOf(f: PortFact): Provenance["dateRole"] {
  if (f.basis === "official") return f.kind === "weather" ? "issued" : "checked";
  if (f.kind === "test") return "sampled";
  if (f.basis === "measured") return "measured";
  if (f.kind === "tide") return "predicted";
  return "issued";
}

/** A short source for the stamp line: "CDFW", "NOAA NDBC", "CDPH". */
export function shortSource(name: string): string {
  if (/CDPH/.test(name)) return "CDPH";
  if (/CDFW/.test(name)) return name.includes("RAMP") ? "CDFW RAMP" : "CDFW";
  if (/National Weather Service|^NWS/.test(name)) return "NWS";
  return name.replace(/\s*\(.*\)$/, "");
}

/** A PortFact as a stamp's provenance. `title` names the reading for the sheet's heading. */
export function provenanceOf(f: PortFact, title: string, extra: Partial<Provenance> = {}): Provenance | null {
  if (!f.as_of) return null;
  return {
    kind: kindOfFact(f),
    title,
    value: f.value ?? undefined,
    source: shortSource(f.source.name),
    sourceLong: f.source.name,
    sourceUrl: f.source.url && /^https?:/.test(f.source.url) ? f.source.url : undefined,
    date: f.as_of,
    dateRole: roleOf(f),
    caveats: f.detail ? [f.detail] : undefined,
    ...extra,
  };
}
