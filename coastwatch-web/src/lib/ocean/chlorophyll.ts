import { PromiseLru } from "@/lib/ocean/cache";
import { chlEstimateLabel, type ChlEstimateIndex, type ChlVar, type OceanGrid, type OceanSource } from "@/lib/ocean/contract";
import { decodeDods } from "@/lib/ocean/dap";
import { gridRecord, gzipCodes, isFill, packCodes, PACK_CHL_LOG10, PACK_MASK, shoreMask } from "@/lib/ocean/pack";
import { errMsg, getBytes, getText, oceanNow } from "@/lib/ocean/upstream";

/**
 * The "chlorophyll.estimate" family: a gap-filled Level-4 chlorophyll field, read live and cached (nothing is
 * published). The source sits behind `ChlEstimateSource` so NOAA DINEOF (today: no account, CC0, but no per-pixel
 * uncertainty, no observed/filled flag and about 3× low within 1 km of shore) can be replaced by Copernicus
 * GlobColour gap-free 4 km (CHL, CHL_uncertainty, flags INTERPOLATED) once the owner sets up an account; a
 * GlobColour source would also return `uncertainty` and `filled` grids. PR #74's staging estimate is not used.
 */
type Geometry = Pick<OceanGrid, "width" | "height" | "lat_first" | "lat_step" | "lon_first" | "lon_step">;

export type ChlField = { geometry: Geometry; values: ArrayLike<number>; fill: number };

export interface ChlEstimateSource {
  id: string;
  source: OceanSource;
  /** Short name for the label, e.g. "NOAA DINEOF". */
  short: string;
  perPixelUncertainty: boolean;
  filledFlag: boolean;
  nearShoreKm: number;
  /** Days after which the newest day is called stale. */
  staleAfterDays: number;
  caveats: string[];
  /** The newest available day, YYYY-MM-DD. */
  latestDate(): Promise<string>;
  /** The California field for a day, in mg/m³. */
  field(date: string): Promise<ChlField>;
}

export const ERDDAP = "https://coastwatch.noaa.gov/erddap/griddap";
export const DINEOF_ID = "noaacwNPPN20S3ASCIDINEOF2kmDaily";
/** The California box (north to south: the dataset's latitude axis descends). */
const BOX = { north: "42.1", south: "32.4", west: "-125.5", east: "-117.0" };

export const dineofLatestUrl = () => `${ERDDAP}/${DINEOF_ID}.csv?time%5B(last)%5D`;
export const dineofFieldUrl = (date: string) =>
  `${ERDDAP}/${DINEOF_ID}.dods?chlor_a%5B(${date}T12:00:00Z)%5D%5B(0.0)%5D%5B(${BOX.north}):1:(${BOX.south})%5D%5B(${BOX.west}):1:(${BOX.east})%5D`;

/** The geometry of a 1-D latitude and longitude axis pair (float32 axes rounded to a clean step). */
export function axisGeometry(lat: ArrayLike<number>, lon: ArrayLike<number>): Geometry {
  const r = (x: number) => Math.round(x * 1e6) / 1e6;
  return {
    width: lon.length,
    height: lat.length,
    lat_first: r(lat[0]),
    lat_step: r((lat[lat.length - 1] - lat[0]) / Math.max(1, lat.length - 1)),
    lon_first: r(lon[0]),
    lon_step: r((lon[lon.length - 1] - lon[0]) / Math.max(1, lon.length - 1)),
  };
}

export function decodeDineof(body: Uint8Array): ChlField {
  const v = decodeDods(body);
  const chl = v.get("chlor_a"), lat = v.get("latitude"), lon = v.get("longitude");
  if (!chl || !lat || !lon) throw new Error("DINEOF response is missing chlor_a or its axes");
  if (chl.data.length !== lat.data.length * lon.data.length) throw new Error("DINEOF response has an unexpected grid size");
  return { geometry: axisGeometry(lat.data, lon.data), values: chl.data, fill: -999 };
}

export const DINEOF: ChlEstimateSource = {
  id: "noaa-dineof-2km",
  source: {
    name: "NOAA CoastWatch DINEOF gap-filled chlorophyll (VIIRS S-NPP, NOAA-20 and Sentinel-3A OLCI), 2 km daily",
    product: DINEOF_ID,
    url: `${ERDDAP}/${DINEOF_ID}.html`,
    info_url: "https://doi.org/10.1109/tgrs.2023.3271465",
    licence: "CC0-1.0",
    developmental: false,
  },
  short: "NOAA DINEOF",
  perPixelUncertainty: false,
  filledFlag: false,
  nearShoreKm: 2,
  staleAfterDays: 4,
  caveats: [
    "Reconstructed by DINEOF (Liu & Wang 2023) where satellites saw nothing: an estimate, not an observation.",
    "No per-pixel uncertainty and no flag separating observed from filled cells.",
    "Reads about 3× low within about 1 km of shore (median log10 bias −0.48 against OLCI); cells within about 2 km of shore are flagged near shore.",
    "Chlorophyll is not toxin; this is not a bloom or toxin prediction.",
  ],
  async latestDate() {
    const csv = await getText(dineofLatestUrl(), { timeoutMs: 8000, revalidate: 21600 });
    const m = /(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}Z/.exec(csv);
    if (!m) throw new Error("DINEOF time list could not be read");
    return m[1];
  },
  async field(date) {
    return decodeDineof(await getBytes(dineofFieldUrl(date), { timeoutMs: 15000, revalidate: 86400 }));
  },
};

export const gridPath = (date: string, v: ChlVar) => `/explore/model/chlorophyll/grid?date=${date}&var=${v}`;

const dayDiff = (date: string, now: Date) => Math.floor((Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) - Date.parse(`${date}T00:00:00Z`)) / 86400e3);

export function buildChlIndex(src: ChlEstimateSource, date: string, geometry: Geometry, now: Date, reason: string | null): ChlEstimateIndex {
  const age = dayDiff(date, now);
  const stale = age > src.staleAfterDays;
  const why = [
    stale ? `The newest ${src.short} day is ${date}, ${age} days old; no newer day has been published.` : null,
    reason,
  ].filter(Boolean);
  return {
    status: stale ? "stale" : "current",
    reason: why.length ? why.join(" ") : null,
    family: "chlorophyll.estimate",
    track: "estimates",
    source: src.source,
    date,
    age_days: age,
    units: "mg/m³",
    label: chlEstimateLabel(date, src.short, src.perPixelUncertainty),
    grids: { chl: gridRecord(geometry, PACK_CHL_LOG10, gridPath(date, "chl")), mask: gridRecord(geometry, PACK_MASK, gridPath(date, "mask")) },
    per_pixel_uncertainty: src.perPixelUncertainty,
    filled_flag: src.filledFlag,
    near_shore_km: src.nearShoreKm,
    caveats: src.caveats,
    generated_at: now.toISOString(),
  };
}

export type PackedChl = { geometry: Geometry } & Record<ChlVar, Buffer>;

export function packChl(f: ChlField, nearShoreKm: number): PackedChl {
  const codes = packCodes(f.values, PACK_CHL_LOG10, f.fill);
  return {
    geometry: f.geometry,
    chl: gzipCodes(codes),
    mask: gzipCodes(shoreMask((k) => !isFill(f.values[k], f.fill), f.geometry, nearShoreKm)),
  };
}

let source: ChlEstimateSource = DINEOF;
/** Tests (or a future GlobColour source) swap the source here. */
export function setChlSource(s: ChlEstimateSource) {
  source = s;
  resetChlCaches();
}

const fields = new PromiseLru<PackedChl>(4);
let latest: { at: number; ttl: number; date: string | null; err: string | null } | null = null;
let lastGood: string | null = null;

export const isDate = (s: string) => /^\d{4}-\d{2}-\d{2}$/.test(s) && !Number.isNaN(Date.parse(`${s}T00:00:00Z`));

/** One day's packed grids (fetched once, then cached). Days more than 60 days old or in the future are refused. */
export async function chlField(date: string): Promise<PackedChl> {
  if (!isDate(date)) throw new RangeError("date must be YYYY-MM-DD");
  const age = dayDiff(date, await oceanNow());
  if (age < 0 || age > 60) throw new RangeError("date outside the served window");
  const s = source;
  return fields.get(`${s.id}/${date}`, async () => packChl(await s.field(date), s.nearShoreKm));
}

/** The estimate index: newest day (cached 6 h; 10 min after an error, serving the last good day). */
export async function chlEstimateIndex(): Promise<ChlEstimateIndex> {
  const now = await oceanNow();
  const wall = Date.now();
  if (!latest || wall - latest.at > latest.ttl) {
    try {
      const date = await source.latestDate();
      latest = { at: wall, ttl: 21600e3, date, err: null };
      lastGood = date;
    } catch (e) {
      latest = { at: wall, ttl: 600e3, date: null, err: errMsg(e) };
    }
  }
  const date = latest.date ?? lastGood;
  const fallback = latest.err && date ? `The newest day could not be checked (${latest.err}); the last good day is shown.` : null;
  if (date) {
    try {
      const f = await chlField(date); // the index carries the real grid geometry, so the field is read once here
      return buildChlIndex(source, date, f.geometry, now, fallback);
    } catch (e) {
      return unavailableChl(`${source.short} could not be read: ${errMsg(e)}`, now);
    }
  }
  return unavailableChl(`${source.short} could not be read: ${latest.err ?? "no day found"}`, now);
}

export function unavailableChl(reason: string, now: Date): ChlEstimateIndex {
  return {
    status: "unavailable", reason, family: "chlorophyll.estimate", track: "estimates", source: source.source, date: null, age_days: null,
    units: "mg/m³", label: null, grids: null, per_pixel_uncertainty: source.perPixelUncertainty, filled_flag: source.filledFlag,
    near_shore_km: source.nearShoreKm, caveats: source.caveats, generated_at: now.toISOString(),
  };
}

export const chlSource = () => source;

/** Tests only. */
export function resetChlCaches() {
  fields.clear();
  latest = null;
  lastGood = null;
}
