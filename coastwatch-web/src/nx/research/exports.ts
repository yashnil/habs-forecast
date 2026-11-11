/**
 * Research exports, DOM-free. Every file opens with `#` lines saying what the numbers are (kind, units,
 * source, request, licence, pipeline run, how an empty value reads), as v11's did; every row carries its
 * own source columns, so a row copied out of the file still says where it came from. Values are written
 * exactly as published: nothing filled, interpolated or rounded.
 */
import type { ObservationDataset, ObsStation } from "@/generated/observations";
import { day, range } from "@/nx/ds/format";
import { dayKey, inWin, LABEL, UNIT, type Strip } from "./model";

const DISCLAIMER = "# Water samples are not seafood tests. Independent research preview; not affiliated with NOAA, CDFW or CDPH.";
const line = (s: string) => `# ${s.replace(/\s*\n\s*/g, " ")}`;
const cell = (v: number | string | null | undefined) => (v == null ? "" : typeof v === "number" ? String(v) : /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
const table = (head: string[], cols: string[], body: (number | string | null)[][]) => [...head, cols.join(","), ...body.map((r) => r.map(cell).join(","))].join("\n") + "\n";

export const SAMPLE_COLUMNS = ["sample_time_utc", "variable", "value", "units", "qualifier", "kind", "station_id", "station_name", "latitude", "longitude", "source", "source_dataset", "source_url"] as const;

const QUALIFIER = { zero: "reported_zero", rejected: "rejected_negative" } as const;

/**
 * Every sample of every measured strip inside the window, one row per sample and quantity, in time order. A
 * reported 0 keeps its 0 and its qualifier; a rejected sample has no value; a visit that did not measure a
 * quantity has no row for it (not measured is never zero).
 */
export function windowCsv(ds: Pick<ObservationDataset, "generated_at" | "provenance">, st: ObsStation, strips: Strip[], win: [number, number]): string {
  const measured = strips.filter((s) => s.kind === "measured" && s.v);
  const p = ds.provenance;
  const head = [
    `# Neritic research export: ${st.name} (${st.station_id}), samples ${dayKey(win[0])} to ${dayKey(win[1])} (UTC days)`,
    line("Kind: measured (shore-station water samples), not a forecast and not a seafood test"),
    line(`Quantities: ${measured.map((s) => `${s.v} = ${LABEL[s.v!]} (${UNIT[s.v!]})`).join("; ")}`),
    line(`Source: ${p.source_name}; ${st.source_url}`),
    ...(st.request_url ? [line(`Request: ${st.request_url}`)] : []),
    ...(st.retrieved_at ? [line(`Retrieved by Neritic: ${st.retrieved_at}`)] : []),
    line(`Licence: ${p.license}`),
    ...(p.citation ? [line(`Citation: ${p.citation}`)] : []),
    line(`Pipeline run: ${p.pipeline_run_id} (version ${p.pipeline_version}); dataset generated ${ds.generated_at}`),
    "# Empty value = no value (rejected by QC). A quantity a visit did not measure has no row: not measured is never zero.",
    "# qualifier: reported_zero = the lab reported 0, at or below an unpublished limit, not absent; flag_high = above the plausibility ceiling, kept as published; rejected_negative = a negative value, rejected.",
    DISCLAIMER,
  ];
  const rows: (number | string | null)[][] = [];
  for (const s of measured) {
    for (const pt of s.points) {
      if (!inWin(pt.t, win)) continue;
      const q = pt.cls === "value" ? (pt.flagged ? "flag_high" : null) : QUALIFIER[pt.cls];
      rows.push([pt.time, s.v, pt.cls === "rejected" ? null : pt.v, UNIT[s.v!], q, "measured", st.station_id, st.name, st.lat, st.lon, "CalHABMAP via SCCOOS ERDDAP", st.station_id, st.source_url]);
    }
  }
  rows.sort((a, b) => String(a[0]).localeCompare(String(b[0])) || measured.findIndex((s) => s.v === a[1]) - measured.findIndex((s) => s.v === b[1]));
  return table(head, [...SAMPLE_COLUMNS], rows);
}

export const MODEL_COLUMNS = ["date", "variable", "probability", "n_cells", "partial", "kind", "station_id", "radius_km", "source"] as const;

/** The model strip's days inside the window: model probabilities, never toxin measurements. */
export function modelCsv(st: ObsStation, strip: Strip, win: [number, number], variable: string): string {
  const radius = st.charm?.radius_km ?? 15;
  const head = [
    `# Neritic research export: ${st.name} (${st.station_id}), C-HARM nowcast near the station, ${variable}, ${dayKey(win[0])} to ${dayKey(win[1])}`,
    line(`Kind: model (NOAA C-HARM v3.1 nowcast). ${strip.label}: the median of the valued model cells within ${radius} km of the station that day; not a value at the station`),
    line("Probability of conditions in the water, not toxin in seafood. Never compare it numerically with the samples."),
    "# partial = fewer than 60% of the usual number of cells had a value that day (a display flag; the value is unchanged)",
    "# Days without a published run have no row (never interpolated).",
    DISCLAIMER,
  ];
  const rows = strip.points.filter((p) => inWin(p.t, win)).map((p) => [p.time, variable, +((p.v as number) / 100).toFixed(6), p.n ?? null, strip.partial?.has(p.t) ? "yes" : "no", "model", st.station_id, radius, "NOAA C-HARM v3.1 via CoastWatch ERDDAP"]);
  return table(head, [...MODEL_COLUMNS], rows);
}

/** One line to cite the window: the program, the station, the window, when it was read, and the link back. */
export function citation(ds: Pick<ObservationDataset, "provenance">, st: ObsStation, strips: Strip[], win: [number, number], href: string): string {
  const qs = strips.filter((s) => s.kind === "measured" && s.v).map((s) => s.v);
  const base = ds.provenance.citation ?? "California Harmful Algal Bloom Monitoring and Alert Program (CalHABMAP), via SCCOOS ERDDAP.";
  const read = st.retrieved_at ? ` Retrieved ${day(st.retrieved_at, { year: true })}.` : "";
  return `${base} ${st.name} (${st.station_id}): ${qs.join(", ")}, samples ${range(win[0], win[1], { year: true })}.${read} Viewed with Neritic: ${href}`;
}
