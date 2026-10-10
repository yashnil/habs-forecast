import type { ForecastRun, LayerArtifact } from "@/generated/schema";
import { hoursAgo } from "@/lib/currents";
import { formatDate, formatDateTimePT } from "@/lib/time";

export type Stamp = { kind: "observation" | "model"; text: string; testid: string };
const short = (d: string) => formatDate(d).replace(/^\w+, /, ""); // "Oct 8"

/** One line per layer on the map: what it is and when it was observed or is valid for. */
export function stampLines(p: {
  group: "forecast" | "satellite" | "currents";
  forecast?: LayerArtifact | null;
  run?: ForecastRun | null;
  satellite?: LayerArtifact | null;
  imagery?: LayerArtifact | null;
  currents?: LayerArtifact | null;
  combined?: boolean;
  now?: Date | null;
}): Stamp[] {
  const out: Stamp[] = [];
  if (p.group === "forecast" && p.forecast?.time.valid_date) {
    const lead = p.forecast.time.lead_days ?? 0;
    out.push({
      kind: "model",
      testid: "stamp-forecast",
      text: `C-HARM ${lead === 0 ? "nowcast" : "forecast"} for ${formatDate(p.forecast.time.valid_date)}${p.run ? ` · issued ${short(p.run.issued_date)}` : ""}`,
    });
  }
  const sat = p.satellite;
  if ((p.group === "satellite" || p.combined) && sat) {
    const ms = sat.multisensor;
    const comp = sat.composite;
    const text = ms
      ? `Chlorophyll, multi-sensor · newest pixel ${short(sat.time.observed_date ?? "")}; each pixel has its own date`
      : comp
        ? `Chlorophyll, Sentinel-3 300 m · pixels observed ${short(comp.oldest_observed_date)}–${short(comp.newest_observed_date)}`
        : sat.time.observed_date
          ? `Chlorophyll, Sentinel-3 300 m · overpass ${formatDate(sat.time.observed_date)}`
          : "Chlorophyll";
    out.push({ kind: "observation", testid: "stamp-satellite", text });
  }
  if (p.group === "satellite" && p.imagery?.time.observed_date && !sat) {
    out.push({ kind: "observation", testid: "stamp-imagery", text: `NASA GIBS imagery · ${formatDate(p.imagery.time.observed_date)}` });
  }
  const c = p.currents;
  if (p.group === "currents" && c) {
    const mean = c.layer_id.endsWith("mean24h");
    const t = mean ? (c.time.observed_times ?? []).at(-1) : c.time.valid_time;
    out.push({
      kind: "observation",
      testid: "stamp-currents",
      text: t
        ? `Currents, HF radar · ${mean ? `24 h mean to ${formatDateTimePT(t)}` : `observed ${formatDateTimePT(t)}`}${p.now ? ` (${hoursAgo(t, p.now)} h ago)` : ""}`
        : "Currents, HF radar",
    });
  }
  return out;
}
