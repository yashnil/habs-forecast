import type { FreshnessPolicy, LayerArtifact, Manifest, ValueGrid } from "@/generated/schema";
import { CHARM_LEADS, charmLayer, charmRun, satelliteDays, satelliteLatest } from "@/lib/layers";
import { chooseHour, currentsHourly, currentsMean } from "@/lib/currents";
import { derivedOutlook, derivedSst, sstIsAnalysis, type DerivedGrid, type DerivedIndex } from "@/lib/derived";
import { classifyDate, type FreshnessState } from "@/lib/freshness";
import { allPast, isPastRun } from "@/lib/modelTime";
import type { StationLite } from "@/lib/data";
import { day, pacificYmd, range, time } from "@/nx/ds/format";
import { gridBounds, type FieldGrid } from "@/nx/chart/layers/grid";

/**
 * What Explore shows, built on the server from the published manifest and the derived index (no field is added
 * to either: the demo and deployed web reject unknown manifest fields). Every frame is a real published frame,
 * stamped with its own dates; availability is the published coverage of each day or hour; a model run whose
 * every lead is before today is a past run (T3). Small enough to pass to the client as props (no values, only
 * where to read them).
 */
export type ProductId = "chl" | "sst" | "charm" | "outlook";
export type FrameKind = "measured" | "analysis" | "model" | "outlook";

export type GridSpec = FieldGrid & {
  /** a single-file grid (relative to the data base) */
  url: string | null;
  chunks: { rows: number; cols: number; url_template: string; present: string[] } | null;
};

export type ExploreFrame = {
  id: string;
  product: ProductId;
  /** the time track the frame sits on (a product may have two sensors) */
  track: string;
  /** short, on the rail: "Oct 6", "Latest" */
  label: string;
  /** the full stamp: what the frame's dates are */
  stamp: string;
  /** the frame's span (Pacific days); a single day has start = end */
  start: string;
  end: string;
  kind: FrameKind;
  /** a model frame valid before today (T3) */
  past: boolean;
  /** share of the coastal ocean with a value, 0..1, where published */
  volume: number | null;
  /** T17, from the layer's own freshness policy */
  fresh: FreshnessState;
  grid: GridSpec | null;
  /** latest views: each pixel's age in days before `referenceDate` */
  ageGrid?: GridSpec | null;
  referenceDate?: string | null;
  /** a model run's or an outlook's issue date (a readout dates an outlook by its window and this, never by its last day) */
  issued?: string | null;
  /** outlook: each cell's usual value for the period (equal means "no signal") */
  typicalGrid?: GridSpec | null;
  sensor: string;
};

export type Track = {
  id: string;
  product: ProductId;
  label: string;
  /** each published day's coverage (data volume), including days with nothing seen */
  days: { date: string; volume: number }[];
};

export type ExploreProduct = {
  id: ProductId;
  /** the legend's title */
  name: string;
  unit: string;
  kind: FrameKind;
  source: string;
  sourceUrl: string | null;
  frames: ExploreFrame[];
  tracks: Track[];
  defaultFrame: string;
  /** every lead before today (C-HARM) */
  pastRun?: boolean;
  runIssued?: string | null;
};

export type CurrentFrame = {
  id: string;
  label: string;
  stamp: string;
  date: string;
  time: string;
  mean: boolean;
  volume: number | null;
  fresh: FreshnessState;
  u: GridSpec;
  v: GridSpec;
};

/**
 * `latest`: the station's own newest measured value per variable, in the shape Research's `stationsMeasuring`
 * reads (src/lib/observations.ts). The slim list knows particulate domoic acid and the two Pseudo-nitzschia
 * groups only; for any other variable a station is "not known to measure it", never assumed to.
 */
export type ExploreStation = { id: string; name: string; lat: number; lon: number; last: string | null; latest: { variable: string; value: number | null; date: string; n_last_365d: number }[] };

/** A layer the product offers but this data run does not hold: it stays in the switch, disabled, with the reason. */
export type MissingLayer = { id: ProductId; label: string; why: string };

export type ExploreModel = {
  now: string;
  today: string;
  baseUrl: string;
  products: ExploreProduct[];
  /** layers absent from this run (never silently dropped) */
  missing: MissingLayer[];
  currents: { frames: CurrentFrame[]; defaultFrame: string; source: string; sourceUrl: string | null; days: { date: string; volume: number; hours: number }[] } | null;
  stations: ExploreStation[];
  /** the rail's span, Pacific days [first, last] */
  axis: [string, string];
  /** where the observed chlorophyll has values: [west, south, east, north] */
  extent: [number, number, number, number] | null;
};

export function gridSpec(g: ValueGrid | DerivedGrid | null | undefined): GridSpec | null {
  if (!g) return null;
  const ch = g.chunks ?? null;
  return {
    width: g.width,
    height: g.height,
    lat_first: g.lat_first,
    lat_step: g.lat_step,
    lon_first: g.lon_first,
    lon_step: g.lon_step,
    scale_factor: g.scale_factor,
    add_offset: g.add_offset,
    nodata: g.nodata ?? 65535,
    transform: g.transform === "log10" ? "log10" : "none",
    url: ch ? null : (g as { url?: string }).url ?? null,
    chunks: ch ? { rows: ch.rows, cols: ch.cols, url_template: ch.url_template, present: [...ch.present] } : null,
  };
}

const addDays = (d: string, n: number) => new Date(Date.parse(`${d}T12:00:00Z`) + n * 86_400_000).toISOString().slice(0, 10);
const coverage = (l: LayerArtifact): number | null => l.coverage?.domain_observed_fraction ?? l.qc?.valid_fraction ?? null;
const freshOf = (policy: FreshnessPolicy | null | undefined, date: string | null, now: Date): FreshnessState => (policy ? classifyDate(policy, date, now).state : "unavailable");
const sensorName = (l: LayerArtifact, fallback: string) => (l.platforms?.length ? l.platforms.join(", ") : fallback);

function chlProduct(m: Manifest, now: Date): ExploreProduct | null {
  const frames: ExploreFrame[] = [];
  const tracks: Track[] = [];
  const olci = satelliteLatest(m, "olci300");
  const days = satelliteDays(m, "olci300");
  if (olci?.grid) {
    const c = olci.composite;
    const a = c?.oldest_observed_date ?? olci.time.observed_date ?? "";
    const b = c?.newest_observed_date ?? olci.time.observed_date ?? a;
    frames.push({
      id: "olci:latest",
      product: "chl",
      track: "olci",
      label: "Latest",
      stamp: a && b && a !== b ? `pixels ${range(a, b)}, each its own latest day seen` : `seen ${day(b)}`,
      start: a,
      end: b,
      kind: "measured",
      past: false,
      volume: coverage(olci),
      fresh: freshOf(olci.freshness, b, now),
      grid: gridSpec(olci.grid),
      ageGrid: gridSpec(c?.age_grid),
      referenceDate: c?.reference_date ?? null,
      sensor: "Sentinel-3 OLCI, 300 m",
    });
  }
  for (const d of days) {
    const date = d.time.observed_date;
    if (!date || !d.grid) continue;
    frames.push({ id: `olci:${date}`, product: "chl", track: "olci", label: day(date), stamp: `seen ${day(date)}`, start: date, end: date, kind: "measured", past: false, volume: coverage(d), fresh: freshOf(d.freshness, date, now), grid: gridSpec(d.grid), sensor: "Sentinel-3 OLCI, 300 m" });
  }
  if (olci || days.length) tracks.push({ id: "olci", product: "chl", label: "OLCI 300 m", days: days.filter((d) => d.time.observed_date).map((d) => ({ date: d.time.observed_date!, volume: d.grid ? (coverage(d) ?? 0) : 0 })) });
  const viirs = satelliteLatest(m, "viirs750");
  if (viirs?.grid) {
    const c = viirs.composite;
    const a = c?.oldest_observed_date ?? viirs.time.observed_date ?? "";
    const b = c?.newest_observed_date ?? viirs.time.observed_date ?? a;
    const sensor = `VIIRS (${sensorName(viirs, "VIIRS").replace(/^VIIRS,?\s*/, "") || "NOAA"}), 750 m`;
    frames.push({ id: "viirs:latest", product: "chl", track: "viirs", label: "Latest", stamp: a !== b ? `pixels ${range(a, b)}, each its own latest day seen` : `seen ${day(b)}`, start: a, end: b, kind: "measured", past: false, volume: coverage(viirs), fresh: freshOf(viirs.freshness, b, now), grid: gridSpec(viirs.grid), ageGrid: gridSpec(c?.age_grid), referenceDate: c?.reference_date ?? null, sensor });
    tracks.push({ id: "viirs", product: "chl", label: "VIIRS 750 m", days: (c?.days ?? []).map((d) => ({ date: d.date, volume: d.pixels_used > 0 ? 1 : 0 })) });
  }
  if (!frames.length) return null;
  const src = olci ?? viirs!;
  return { id: "chl", name: "Chlorophyll", unit: "mg/m³", kind: "measured", source: olci ? "Sentinel-3 OLCI via NOAA CoastWatch" : "VIIRS via NOAA CoastWatch", sourceUrl: src.provenance?.source_url ?? null, frames, tracks, defaultFrame: frames[0].id };
}

function charmProduct(m: Manifest, today: string, now: Date): ExploreProduct | null {
  const variable = "particulate_domoic";
  const layers = CHARM_LEADS.map((l) => charmLayer(m, variable, l));
  const run = charmRun(m);
  const frames: ExploreFrame[] = [];
  layers.forEach((l, lead) => {
    const valid = l?.time.valid_date;
    if (!l?.grid || !valid) return;
    frames.push({
      id: `charm:${lead}`,
      product: "charm",
      track: "charm",
      label: day(valid),
      stamp: `valid ${day(valid)}, run of ${day(run?.issued_date ?? l.time.issued_date ?? valid)}`,
      start: valid,
      end: valid,
      kind: "model",
      past: isPastRun(valid, valid, today),
      volume: 1,
      fresh: freshOf(l.freshness, l.time.issued_date ?? null, now),
      grid: gridSpec(l.grid),
      issued: run?.issued_date ?? l.time.issued_date ?? null,
      sensor: "C-HARM v3.1",
    });
  });
  if (!frames.length) return null;
  const pastRun = allPast(frames.map((f) => f.end), today);
  // open on the lead valid today, else the next one after today, else (a past run) its last lead
  const pick = frames.find((f) => f.start === today) ?? frames.find((f) => f.start > today) ?? frames[frames.length - 1];
  const l0 = layers.find(Boolean)!;
  return {
    id: "charm",
    name: "C-HARM, P(pDA > 500 ng/L)",
    unit: "probability",
    kind: "model",
    source: "NOAA C-HARM v3.1",
    sourceUrl: l0.provenance?.source_url ?? null,
    frames,
    tracks: [{ id: "charm", product: "charm", label: "C-HARM", days: frames.map((f) => ({ date: f.start, volume: 1 })) }],
    defaultFrame: pick.id,
    pastRun,
    runIssued: run?.issued_date ?? l0.time.issued_date ?? null,
  };
}

function outlookProduct(d: DerivedIndex | null, today: string): ExploreProduct | null {
  const o = derivedOutlook(d);
  if (!o) return null;
  const typ = o.typical_grid && o.typical_grid.width === o.grid.width && o.typical_grid.height === o.grid.height ? gridSpec(o.typical_grid) : null;
  // T3: once every day of its window has passed, the outlook is a past run, worded and drawn like C-HARM's
  const past = isPastRun(o.valid_from, o.valid_to, today);
  const frame: ExploreFrame = { id: "outlook", product: "outlook", track: "outlook", label: range(o.valid_from, o.valid_to), stamp: `mean of ${range(o.valid_from, o.valid_to)}, issued ${day(o.issued)}`, start: o.valid_from, end: o.valid_to, kind: "outlook", past, volume: null, fresh: past ? "historical" : "current", grid: gridSpec(o.grid), issued: o.issued, typicalGrid: typ, sensor: "Neritic outlook" };
  return { id: "outlook", name: "Chlorophyll outlook", unit: "mg/m³", kind: "outlook", source: "Neritic, from NASA MODIS-Aqua 8-day", sourceUrl: null, frames: [frame], tracks: [{ id: "outlook", product: "outlook", label: "Outlook", days: [] }], defaultFrame: frame.id, pastRun: past, runIssued: o.issued };
}

/**
 * The temperature's freshness limits (T17). The derived index publishes no policy for it yet, so one the layer
 * carries wins; otherwise the daily analysis's own cadence: MUR is daily with about two days' latency
 * (E-science-and-perf.md §1), so three days is current and a week is the stale limit.
 */
export const SST_FRESHNESS: FreshnessPolicy = { basis: "observed_date", current_max_age_days: 3, stale_max_age_days: 7, note: "JPL MUR is a daily analysis published about two days behind." } as FreshnessPolicy;

function sstPolicy(s: object): FreshnessPolicy {
  const f = (s as { freshness?: Partial<FreshnessPolicy> | null }).freshness;
  return f && typeof f.current_max_age_days === "number" && typeof f.stale_max_age_days === "number" ? (f as FreshnessPolicy) : SST_FRESHNESS;
}

/** A source's short name for a legend line: the product, without its expansion or its route ("…, via NOAA CoastWatch …"). */
export const shortSensor = (source: string) => source.replace(/\s*\([^)]*\)/g, "").replace(/,?\s+via\s.*$/i, "").trim();

function sstProduct(d: DerivedIndex | null, now: Date): ExploreProduct | null {
  const s = derivedSst(d);
  if (!s) return null;
  const date = s.observed_dates[s.observed_dates.length - 1];
  const kind: FrameKind = sstIsAnalysis(s) ? "analysis" : "measured";
  const frame: ExploreFrame = { id: "sst", product: "sst", track: "sst", label: day(date), stamp: kind === "analysis" ? `analysis for ${day(date)}` : `seen ${day(date)}`, start: date, end: date, kind, past: false, volume: null, fresh: freshOf(sstPolicy(s), date, now), grid: gridSpec(s.grid), sensor: shortSensor(s.source) };
  return { id: "sst", name: "Sea temperature", unit: "°C", kind, source: s.source, sourceUrl: null, frames: [frame], tracks: [{ id: "sst", product: "sst", label: "Temperature", days: [{ date, volume: 1 }] }], defaultFrame: frame.id };
}

function currentsModel(m: Manifest, now: Date): ExploreModel["currents"] {
  const hours = currentsHourly(m);
  const mean = currentsMean(m);
  const frames: CurrentFrame[] = [];
  for (const l of hours) {
    const t = l.time.valid_time;
    if (!t || !l.vectors) continue;
    frames.push({ id: l.layer_id, label: time(t), stamp: `${day(t)}, ${time(t)}`, date: pacificYmd(t), time: t, mean: false, volume: coverage(l), fresh: freshOf(l.freshness, l.time.observed_date ?? null, now), u: gridSpec(l.vectors.u_grid)!, v: gridSpec(l.vectors.v_grid)! });
  }
  if (!frames.length) return null;
  if (mean?.vectors && mean.time.valid_time) {
    const t = mean.time.valid_time;
    const first = mean.time.observed_times?.[0] ?? t;
    frames.push({ id: mean.layer_id, label: "24 h mean", stamp: `mean of the 24 hours to ${day(t)}, ${time(t)} (from ${time(first)})`, date: pacificYmd(t), time: t, mean: true, volume: coverage(mean), fresh: freshOf(mean.freshness, mean.time.observed_date ?? null, now), u: gridSpec(mean.vectors.u_grid)!, v: gridSpec(mean.vectors.v_grid)! });
  }
  const byDay = new Map<string, { sum: number; hours: number }>();
  for (const f of frames) {
    if (f.mean) continue;
    const e = byDay.get(f.date) ?? { sum: 0, hours: 0 };
    e.sum += f.volume ?? 0;
    e.hours += 1;
    byDay.set(f.date, e);
  }
  const chosen = chooseHour(m).layer;
  return {
    frames,
    defaultFrame: chosen && frames.some((f) => f.id === chosen.layer_id) ? chosen.layer_id : frames.filter((f) => !f.mean).pop()!.id,
    source: "HFRNet HF radar via NOAA CoastWatch",
    sourceUrl: hours[0]?.provenance?.source_url ?? null,
    days: [...byDay].map(([date, e]) => ({ date, volume: e.sum / e.hours, hours: e.hours })),
  };
}

const LAYER_ORDER: ProductId[] = ["chl", "sst", "charm", "outlook"];
const MISSING_LABEL: Record<ProductId, string> = { chl: "Chlorophyll", sst: "Temperature", charm: "C-HARM", outlook: "Outlook" };

export function buildExploreModel({ manifest, derived, stations, baseUrl, now }: { manifest: Manifest; derived: DerivedIndex | null; stations: StationLite[] | null; baseUrl: string; now: Date }): ExploreModel {
  const today = pacificYmd(now);
  const built: Record<ProductId, ExploreProduct | null> = { chl: chlProduct(manifest, now), sst: sstProduct(derived, now), charm: charmProduct(manifest, today, now), outlook: outlookProduct(derived, today) };
  const products = LAYER_ORDER.map((id) => built[id]).filter((p): p is ExploreProduct => !!p);
  const missing: MissingLayer[] = LAYER_ORDER.filter((id) => !built[id]).map((id) => ({ id, label: MISSING_LABEL[id], why: "Not in this data run" }));
  const currents = currentsModel(manifest, now);
  const dates = [...products.flatMap((p) => p.frames.flatMap((f) => [f.start, f.end])), ...products.flatMap((p) => p.tracks.flatMap((t) => t.days.map((d) => d.date))), ...(currents?.frames.map((f) => f.date) ?? [])].filter(Boolean).sort();
  const floor = addDays(today, -16);
  const first = dates.length ? (dates[0] < floor ? floor : dates[0]) : addDays(today, -7);
  const last = [dates[dates.length - 1] ?? today, today].sort()[1];
  const chl = products.find((p) => p.id === "chl");
  const g = chl?.frames[0]?.grid;
  return {
    now: now.toISOString(),
    today,
    baseUrl,
    products,
    missing,
    currents,
    stations: (stations ?? []).map((s) => ({ id: s.station_id, name: s.name, lat: s.lat, lon: s.lon, last: s.latest.map((x) => x.date).sort().pop() ?? null, latest: s.latest.map((x) => ({ variable: x.variable, value: x.value, date: x.date, n_last_365d: x.n_last_365d })) })),
    axis: [first, addDays(last, 1)],
    extent: g ? gridBounds(g) : null,
  };
}
