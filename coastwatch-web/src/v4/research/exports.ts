import type { LayerArtifact, Manifest } from "@/generated/schema";
import type { ObservationDataset, ObsStation, ObsVariable } from "@/generated/observations";
import { samples, unitLabel, type ObsVariableId } from "@/lib/observations";
import { modelOnAxis } from "@/lib/bloom";
import type { CharmVariable } from "@/lib/layers";
import type { AreaCell } from "@/lib/area";
import type { LeadValue } from "@/v4/explore/reads";
import type { TransectCell } from "./transect";

/**
 * Research exports (CSV). Every file opens with `#` lines that say what the numbers are: product
 * class, units, source dataset and its licence, the pipeline run that published them, how an empty
 * value reads, and, for C-HARM, that the numbers are model probabilities and not toxin
 * measurements. Values are written exactly as decoded from the published data: nothing is filled,
 * interpolated or rounded beyond the 16-bit storage the grid already carries.
 */

const DISCLAIMER = "# Not a seafood test. Independent research preview; not affiliated with NOAA, CDFW or CDPH.";
/** one header line: no line breaks inside a `#` comment */
const line = (s: string) => `# ${s.replace(/\s*\n\s*/g, " ")}`;
const cell = (v: number | string | null | undefined) => (v == null ? "" : typeof v === "number" ? String(v) : /[",]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
const rows = (head: string[], cols: string, body: (number | string | null)[][]) => [...head, cols, ...body.map((r) => r.map(cell).join(","))].join("\n") + "\n";

/** The `#` block for C-HARM model output from one or more layers of the same run. */
export function modelHeader(title: string, layers: LayerArtifact[], manifest: Pick<Manifest, "pipeline_run_id" | "pipeline_version" | "generated_at">, extra: string[] = []): string[] {
  const l = layers[0];
  const p = l?.provenance;
  const ids = [...new Set(layers.map((x) => x.provenance.dataset_id).filter(Boolean))];
  return [
    `# Neritic research export: ${title}`,
    line(`Product class: model output (${p?.source_name ?? "C-HARM"}), a probability for each 3 km model cell; not an observation`),
    line(`Variable: ${l?.threshold_text ?? l?.title ?? "—"}; units: ${l?.units ?? "probability (0-1)"}`),
    line(`Source dataset: ${ids.join(", ") || "—"}${p?.product_version ? `, product version ${p.product_version}` : ""} (${p?.institution ?? "—"}); ${p?.source_url ?? ""}`),
    line(`Run issued ${l?.time.issued_date ?? "—"}${l?.time.issued_date_derived ? ` (inferred: ${l.time.issued_date_method ?? "not published upstream"})` : ""}`),
    line(`Licence: ${p?.license ?? "—"}`),
    ...(p?.citation ? [line(`Citation: ${p.citation}`)] : []),
    line(`Pipeline run: ${manifest.pipeline_run_id} (version ${manifest.pipeline_version}), manifest generated ${manifest.generated_at}`),
    ...extra.map(line),
    "# Empty value = no published value (land, nearshore, masked or outside the model); never zero, never interpolated",
    "# Model probability, not a toxin measurement.",
    DISCLAIMER,
  ];
}

/** Station samples in the window (one variable), as published. */
export function stationCsv(ds: Pick<ObservationDataset, "generated_at" | "provenance">, st: ObsStation, v: Pick<ObsVariable, "id" | "units" | "method">, from: number, to: number): string {
  const r = samples(st, v.id as ObsVariableId).filter((s) => s.t >= from && s.t <= to);
  const p = ds.provenance;
  const head = [
    `# Neritic research export: ${st.name} (${st.station_id}), ${v.id} in ${unitLabel(v.units)}`,
    line(`Product class: observation (shore-station water sample), not a forecast`),
    line(`Station position: ${st.lat ?? "—"}°N, ${st.lon ?? "—"}°E; method: ${v.method}`),
    `# Source: CalHABMAP via SCCOOS ERDDAP, ${st.source_url}`,
    ...(st.request_url ? [line(`Request: ${st.request_url}`)] : []),
    line(`Licence: ${p.license}`),
    ...(p.citation ? [line(`Citation: ${p.citation}`)] : []),
    line(`Pipeline run: ${p.pipeline_run_id} (version ${p.pipeline_version})`),
    `# Values as published in Neritic's dataset generated ${ds.generated_at}; empty value = not measured (never zero); qualifier as published (rejected_negative = rejected by QC, no value)`,
    DISCLAIMER,
  ];
  return rows(head, "sample_time_utc,value,qualifier", r.map((s) => [s.time, s.value, s.q]));
}

/** C-HARM nowcast history kept for a station (median of the cells within its radius). */
export function modelHistoryCsv(st: ObsStation, variable: CharmVariable, nowcast: LayerArtifact | null, manifest: Pick<Manifest, "pipeline_run_id" | "pipeline_version" | "generated_at">, method: string | undefined): string {
  const hist = st.charm?.history?.[variable] ?? [];
  const { points } = modelOnAxis(hist, -Infinity, Infinity);
  const partial = new Map(points.map((p) => [p.date, p.partial]));
  const head = modelHeader(`${st.name} (${st.station_id}), C-HARM nowcast history, ${variable}`, nowcast ? [nowcast] : [], manifest, [
    `Each value: the median of the valued nowcast cells within ${st.charm?.radius_km ?? "—"} km of the station that day; n_cells = how many cells had a value${method ? ` (${method})` : ""}`,
    "partial = fewer than 60% of the usual number of cells had a value that day (a Neritic display flag; the value is unchanged)",
    "Shown beside the station's samples, never compared with them as numbers.",
  ]);
  return rows(head, "date,value,n_cells,partial", hist.map((h) => [h.date, h.value, h.n, h.value == null ? null : partial.get(h.date) ? "yes" : "no"]));
}

/** One cell on each day of the run. */
export function cellLeadsCsv(cellAt: { row: number; col: number; lat: number; lon: number }, leads: LeadValue[], layers: LayerArtifact[], manifest: Pick<Manifest, "pipeline_run_id" | "pipeline_version" | "generated_at">, variable: CharmVariable): string {
  const head = modelHeader(`C-HARM cell row ${cellAt.row}, column ${cellAt.col} (centre ${cellAt.lat.toFixed(3)}°N, ${cellAt.lon.toFixed(3)}°E), ${variable}, every day of the run`, layers, manifest, [
    `Quantisation error of the stored values: ± ${layers[0]?.grid?.max_quantization_error ?? "—"}`,
    "Only the days this run published; nothing beyond its horizon.",
  ]);
  return rows(head, "lead_days,kind,valid_date,value", leads.filter((l) => l.issued).map((l) => [l.lead, l.lead === 0 ? "nowcast" : "forecast", l.date, l.value]));
}

/** The cells a transect crosses, in order. */
export function transectCsv(a: { lat: number; lon: number }, b: { lat: number; lon: number }, cells: TransectCell[], layer: LayerArtifact, manifest: Pick<Manifest, "pipeline_run_id" | "pipeline_version" | "generated_at">): string {
  const head = modelHeader(`C-HARM transect, ${layer.variable}, valid ${layer.time.valid_date ?? "—"}`, [layer], manifest, [
    `From ${a.lat.toFixed(4)}°N ${a.lon.toFixed(4)}°E to ${b.lat.toFixed(4)}°N ${b.lon.toFixed(4)}°E; km = distance along the line to where it enters the cell`,
    "Every cell the line crosses is listed once; lat/lon are cell centres (the entry point when outside the domain); in_domain = no means outside the model grid",
  ]);
  return rows(head, "km,row,col,lat,lon,value,in_domain", cells.map((c) => [+c.km.toFixed(3), c.inDomain ? c.row : null, c.inDomain ? c.col : null, +c.lat.toFixed(5), +c.lon.toFixed(5), c.value, c.inDomain ? "yes" : "no"]));
}

/** Every cell whose centre lies inside a box. */
export function areaCsv(bbox: [number, number, number, number], cells: AreaCell[], layer: LayerArtifact, manifest: Pick<Manifest, "pipeline_run_id" | "pipeline_version" | "generated_at">): string {
  const [w, s, e, n] = bbox;
  const head = modelHeader(`C-HARM cells in a box, ${layer.variable}, valid ${layer.time.valid_date ?? "—"}`, [layer], manifest, [
    `Box: ${s.toFixed(4)} to ${n.toFixed(4)}°N, ${w.toFixed(4)} to ${e.toFixed(4)}°E; a cell is inside when its centre is`,
    "Not an average for any harbour.",
  ]);
  return rows(head, "row,col,lat,lon,value", cells.map((c) => [c.row, c.col, +c.lat.toFixed(5), +c.lon.toFixed(5), c.value]));
}

/** Save a CSV in the browser. */
export function download(name: string, text: string) {
  const blob = new Blob([text], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
