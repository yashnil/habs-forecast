import type { Palette, ValueGrid } from "@/generated/schema";
import { cellAt, chunkedCode, decodeCode } from "@/lib/grid";

/**
 * Derived chlorophyll products (v5): an 8-day outlook and a gap-filling estimate, published by the
 * pipeline in their own index, `${dataBase}/derived/index.json`, apart from the manifest (the public
 * v2 site reads the manifest and rejects unknown fields; it never reads this file).
 *
 * Neither product is an observation, and neither is toxin: every place that shows one says what it
 * is ("Outlook, not observed" / "Estimated, not observed"), how it was made, how well it did against
 * plain baselines on held-out data (with n), its uncertainty and its dates. An estimate holds values
 * only where the observed layer has none; observed pixels are drawn above it and never overwritten.
 *
 * A missing or invalid index means the products are simply not offered: nothing else changes.
 */

export type DerivedGrid = ValueGrid & { chunks: NonNullable<ValueGrid["chunks"]> };

export type DerivedSkill = {
  metric: string;
  model: number;
  climatology?: number | null;
  persistence?: number | null;
  test_period?: string;
  n?: number;
  note?: string;
  /** any further baseline the pipeline reports (e.g. climatology_ar1), named by its key */
  [baseline: string]: unknown;
};

export type DerivedValidation = { metric: string; by_band: Record<string, number>; baseline?: string; n?: number; [k: string]: unknown };

type Common = {
  id: string;
  label: string;
  issued: string;
  method: string;
  caveats: string[];
  units?: string;
  grid: DerivedGrid;
  uncertainty_grid?: DerivedGrid | null;
  /** the observed layer an estimate fills (either name; any latest-clear view when absent) */
  observed_layer_id?: string;
  observed_layer?: string;
};
export type OutlookLayer = Common & { kind: "chl_outlook"; valid_from: string; valid_to: string; basis: string; skill: DerivedSkill };
export type EstimateLayer = Common & { kind: "chl_estimate"; based_on_days: string[]; validation: DerivedValidation; basis?: string };
/** Observed (or analysed, e.g. MUR) sea-surface temperature, °C. */
export type SstLayer = Common & { kind: "sst"; observed_dates: string[]; source: string; resolution_km?: number | null };
export type DerivedLayer = OutlookLayer | EstimateLayer | SstLayer;
/**
 * A check of a display rule against measurements, with no grid of its own: e.g.
 * "charm_nearshore_extension", CalHABMAP pier samples against the nearest C-HARM cell (the value
 * the map extends to the shore). `skill` holds the numbers as the pipeline reports them; `demonstrated`
 * (or a note saying there is no skill) decides how the readout words the extension.
 */
export type DerivedValidationEntry = { id: string; kind: "validation"; skill: Record<string, unknown>; note: string; demonstrated?: boolean; caveats?: string[]; /** the pipeline's own sentence for a readout, quoted as is */ readout?: string };
export type DerivedIndex = { version: 1; generated_at: string; layers: DerivedLayer[]; validations?: DerivedValidationEntry[] };

const isStr = (x: unknown): x is string => typeof x === "string" && x.length > 0;
const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const isDate = (x: unknown): x is string => isStr(x) && /^\d{4}-\d{2}-\d{2}/.test(x);

function validGrid(g: unknown): g is DerivedGrid {
  const x = g as Record<string, unknown> | null;
  if (!x || typeof x !== "object") return false;
  const ch = x.chunks as Record<string, unknown> | undefined;
  return (
    x.encoding === "uint16le+gzip" &&
    ["width", "height", "lat_first", "lat_step", "lon_first", "lon_step", "scale_factor", "add_offset"].every((k) => isNum(x[k])) &&
    (x.width as number) > 0 &&
    (x.height as number) > 0 &&
    (x.transform === "log10" || x.transform === "none" || x.transform == null) &&
    !!ch &&
    isNum(ch.rows) &&
    isNum(ch.cols) &&
    isStr(ch.url_template) &&
    Array.isArray(ch.present)
  );
}

/** One layer of the index, checked against the contract; anything malformed is dropped (never half-shown). */
export function validLayer(l: unknown): DerivedLayer | null {
  const x = l as Record<string, unknown> | null;
  if (x && typeof x === "object" && x.kind === "sst") {
    if (!isStr(x.id) || !isStr(x.label) || !isStr(x.source) || !validGrid(x.grid)) return null;
    const dates = Array.isArray(x.observed_dates) ? x.observed_dates.filter(isDate).sort() : [];
    if (!dates.length) return null;
    return {
      ...(x as object),
      kind: "sst",
      issued: isDate(x.issued) ? x.issued : dates[dates.length - 1],
      method: isStr(x.method) ? x.method : "",
      caveats: Array.isArray(x.caveats) ? x.caveats.filter(isStr) : [],
      uncertainty_grid: null,
      observed_dates: dates,
      source: x.source,
      resolution_km: isNum(x.resolution_km) ? x.resolution_km : null,
    } as SstLayer;
  }
  if (!x || typeof x !== "object" || !isStr(x.id) || !isStr(x.label) || !isDate(x.issued) || !isStr(x.method) || !validGrid(x.grid)) return null;
  const caveats = Array.isArray(x.caveats) ? x.caveats.filter(isStr) : [];
  const unc = x.uncertainty_grid != null && validGrid(x.uncertainty_grid) ? (x.uncertainty_grid as DerivedGrid) : null;
  const base = { ...(x as object), caveats, uncertainty_grid: unc } as Common;
  if (x.kind === "chl_outlook") {
    const s = x.skill as Record<string, unknown> | undefined;
    if (!isDate(x.valid_from) || !isDate(x.valid_to) || !isStr(x.basis) || !s || !isStr(s.metric) || !isNum(s.model)) return null;
    // an outlook is offered only with its skill against at least one plain baseline
    if (!isNum(s.climatology) && !isNum(s.persistence)) return null;
    return { ...base, kind: "chl_outlook", valid_from: x.valid_from, valid_to: x.valid_to, basis: x.basis, skill: s as DerivedSkill };
  }
  if (x.kind === "chl_estimate") {
    const v = x.validation as Record<string, unknown> | undefined;
    if (!Array.isArray(x.based_on_days) || !v || !isStr(v.metric) || !v.by_band || typeof v.by_band !== "object") return null;
    const bands = Object.fromEntries(Object.entries(v.by_band as Record<string, unknown>).filter(([, n]) => isNum(n))) as Record<string, number>;
    if (!Object.keys(bands).length) return null;
    return { ...base, kind: "chl_estimate", based_on_days: x.based_on_days.filter(isDate), validation: { ...(v as object), metric: v.metric, by_band: bands } as DerivedValidation };
  }
  return null;
}

/** The index as published, or null when it is missing, unreadable or of another version. */
export function parseDerived(j: unknown): DerivedIndex | null {
  const x = j as Record<string, unknown> | null;
  if (!x || typeof x !== "object" || x.version !== 1 || !Array.isArray(x.layers)) return null;
  const layers = x.layers.map(validLayer).filter((l): l is DerivedLayer => !!l);
  const validations = x.layers
    .filter((l): l is Record<string, unknown> => !!l && typeof l === "object" && (l as Record<string, unknown>).kind === "validation")
    .filter((l) => isStr(l.id) && ((!!l.skill && typeof l.skill === "object") || isStr(l.readout)))
    .map((l) => {
      const skill = (l.skill && typeof l.skill === "object" ? l.skill : {}) as Record<string, unknown>;
      const flag = [l.has_demonstrated_skill, l.demonstrated, skill.has_demonstrated_skill, skill.demonstrated].find((b) => typeof b === "boolean") as boolean | undefined;
      return { id: l.id as string, kind: "validation" as const, skill, note: isStr(l.note) ? l.note : "", demonstrated: flag, caveats: Array.isArray(l.caveats) ? l.caveats.filter(isStr) : [], readout: isStr(l.readout) ? l.readout : undefined };
    });
  return { version: 1, generated_at: isStr(x.generated_at) ? x.generated_at : "", layers, validations };
}

const cache = new Map<string, Promise<DerivedIndex | null>>();

/**
 * `${dataBase}/derived/index.json`, fetched once per base. Never throws: a failure is null. Offline
 * it is not asked for (a field pack does not carry the derived products): they are simply not offered.
 */
export function loadDerived(baseUrl: string): Promise<DerivedIndex | null> {
  if (typeof navigator !== "undefined" && navigator.onLine === false) return Promise.resolve(null);
  const url = `${baseUrl.replace(/\/$/, "")}/derived/index.json`;
  let p = cache.get(url);
  if (!p) {
    p = fetch(url)
      .then((r) => (r.ok ? r.json() : null))
      .then(parseDerived)
      .catch(() => null);
    cache.set(url, p);
  }
  return p;
}

export const derivedOutlook = (d: DerivedIndex | null | undefined): OutlookLayer | null => (d?.layers.find((l): l is OutlookLayer => l.kind === "chl_outlook") ?? null);
export const derivedEstimate = (d: DerivedIndex | null | undefined): EstimateLayer | null => (d?.layers.find((l): l is EstimateLayer => l.kind === "chl_estimate") ?? null);

/** The satellite chlorophyll ramp (pipeline palette "cw-chlorophyll-log-v1"), used when the manifest offers none. */
export const CHL_PALETTE: Palette = {
  id: "cw-chlorophyll-log-v1",
  domain: [-1.30103, 1.69897],
  stops: [
    { value: -1.30103, color: "#102a4c" },
    { value: -0.92603, color: "#14466e" },
    { value: -0.55103, color: "#126483" },
    { value: -0.17603, color: "#147f85" },
    { value: 0.19897, color: "#2a9a7c" },
    { value: 0.57397, color: "#5bb267" },
    { value: 0.94897, color: "#9bc851" },
    { value: 1.32397, color: "#d6dd5e" },
    { value: 1.69897, color: "#f3ef9c" },
  ],
  interpolation: "linear",
  scale: "log10",
  nodata: "transparent",
} as Palette;

export type DerivedRead = {
  /** mg m-3, or null where the product has no value */
  value: number | null;
  /** the typical range from the uncertainty grid (value ÷ and × 10^error), when published */
  range: [number, number] | null;
};

/** The product's value in the cell containing the point (never the nearest, never interpolated), with its typical range. */
export async function derivedAt(baseUrl: string, l: DerivedLayer, lat: number, lon: number): Promise<DerivedRead> {
  const c = cellAt(l.grid, lat, lon);
  if (!c) return { value: null, range: null };
  const v = decodeCode(l.grid, await chunkedCode(baseUrl, l.grid, c.row, c.col));
  if (v == null) return { value: null, range: null };
  let range: [number, number] | null = null;
  const u = l.uncertainty_grid;
  if (u) {
    const uc = cellAt(u, lat, lon);
    // the uncertainty is an expected error in log10 units per cell (decoded with its own transform)
    const e = uc ? decodeCode(u, await chunkedCode(baseUrl, u, uc.row, uc.col)) : null;
    if (e != null && e > 0) range = [v / 10 ** e, v * 10 ** e];
  }
  return { value: v, range };
}

export const derivedSst = (d: DerivedIndex | null | undefined): SstLayer | null => (d?.layers.find((l): l is SstLayer => l.kind === "sst") ?? null);
/** MUR (and other blended, gap-filled products) are analyses, not single observations. */
export const sstIsAnalysis = (l: SstLayer) => /\bmur\b|analys/i.test(`${l.source} ${l.label}`);
export const cToF = (c: number) => (c * 9) / 5 + 32;
/**
 * Sea-surface temperature on a perceptually even ramp (cmocean "thermal": lightness rises steadily
 * from deep blue-black through purple and red to yellow), 8–22 °C, so a front is a visible step
 * in lightness, not only in hue.
 */
export const SST_PALETTE: Palette = {
  id: "neritic-sst-thermal-v1",
  domain: [8, 22],
  stops: [
    { value: 8, color: "#042333" },
    { value: 9.75, color: "#15325a" },
    { value: 11.5, color: "#3e3790" },
    { value: 13.25, color: "#6a4a99" },
    { value: 15, color: "#935b96" },
    { value: 16.75, color: "#bd6a82" },
    { value: 18.5, color: "#e27a5c" },
    { value: 20.25, color: "#f8a53c" },
    { value: 22, color: "#e8fa5b" },
  ],
  interpolation: "linear",
  scale: "linear",
  nodata: "transparent",
} as Palette;

export const nearshoreCheck = (d: DerivedIndex | null | undefined): DerivedValidationEntry | null => d?.validations?.find((v) => v.id === "charm_nearshore_extension") ?? null;

/** Did the check show the extension has skill? false when it says so, or when its note says there is none. */
export function checkShowsSkill(v: DerivedValidationEntry): boolean {
  if (v.demonstrated != null) return v.demonstrated;
  return !/\bno (demonstrated )?skill\b|not (demonstrated|skil)/i.test(v.note);
}

/** "Checked against 214 pier samples (2019–2025): …": the numbers as published, and the note. */
export function checkWords(v: DerivedValidationEntry): string {
  if (v.readout) return v.readout;
  const s = v.skill;
  const n = isNum(s.n) ? s.n : null;
  const nums = Object.entries(s)
    .filter(([k, x]) => isNum(x) && k !== "n")
    .map(([k, x]) => `${k.replace(/_/g, " ")} ${(x as number).toFixed(Number.isInteger(x) ? 0 : 2)}`);
  const head = `Checked against ${n != null ? `${n.toLocaleString("en-US")} ` : ""}CalHABMAP pier sample${n === 1 ? "" : "s"}${isStr(s.period) ? ` (${s.period})` : isStr(s.test_period) ? ` (${s.test_period})` : ""}`;
  return `${head}${isStr(s.metric) ? `, ${s.metric}` : ""}${nums.length ? `: ${nums.join(", ")}` : ""}.${v.note ? ` ${v.note}` : ""}`;
}

/** The check in one sentence, saying "context only" whenever the check found no skill (never twice). */
export function checkSentence(v: DerivedValidationEntry): string {
  const w = checkWords(v);
  return checkShowsSkill(v) || /context only|no demonstrated skill/i.test(w) ? w : `${w} Context only: no demonstrated skill at the shore.`;
}

export const fmtMg = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

/** "RMSE ln(mg m-3)" and friends: lower is better for any error metric. */
export const lowerIsBetter = (metric: string) => /rmse|mae|error|bias/i.test(metric);

/** The skill rows to show: the model, then every baseline the index reports, in a fixed order. */
export function skillRows(s: DerivedSkill): { key: string; label: string; value: number }[] {
  const named: Record<string, string> = { model: "This outlook", climatology_ar1: "Climatology + AR(1)", clim_ar1: "Climatology + AR(1)", climatology: "Climatology", persistence: "Persistence (last view)" };
  const keys = ["model", "climatology_ar1", "clim_ar1", "climatology", "persistence", ...Object.keys(s).filter((k) => !["model", "climatology_ar1", "clim_ar1", "climatology", "persistence", "n", "metric", "test_period", "note"].includes(k))];
  return keys.filter((k, i) => keys.indexOf(k) === i && isNum(s[k])).map((k) => ({ key: k, label: named[k] ?? k.replace(/_/g, " "), value: s[k] as number }));
}

/** Band labels as the pipeline writes them ("0-0.6 km"), with a proper dash. */
export const bandLabel = (b: string) => (b === "bays" ? "Bays" : b.replace(/-/g, "–"));
