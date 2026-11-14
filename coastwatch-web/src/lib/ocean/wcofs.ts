import { PromiseLru } from "@/lib/ocean/cache";
import { cycleName, wcofsLabel, type OceanSource, type WcofsIndex, type WcofsStep, type WcofsVar } from "@/lib/ocean/contract";
import { decodeDods } from "@/lib/ocean/dap";
import { gridRecord, gzipCodes, isFill, packCodes, PACK_MASK, PACK_TEMPERATURE, PACK_VELOCITY, shoreMask } from "@/lib/ocean/pack";
import { errMsg, getBytes, getText, oceanNow } from "@/lib/ocean/upstream";

/**
 * NOAA WCOFS (West Coast Operational Forecast System) surface currents and temperature, read live from the CO-OPS
 * THREDDS OPeNDAP server (docs/coastwatch/v15.1/science/wcofs.md). Each cycle (daily, 03z) has 3-hourly
 * `regulargrid` files on a 0.04° lat/lon grid with currents already rotated to east/north; Neritic reads one
 * California box at depth 0 m, one step per request, and packs it to uint16 grids. Nothing is published.
 *
 * Caches: the cycle index for an hour (15 min while today's cycle has not landed, 5 min after an upstream error,
 * when the last good index is served and marked `fallback`); each step's packed grids in-process (LRU), plus Next's
 * fetch cache and the CDN in front. An outage with nothing cached answers "unavailable", never a substitute.
 */
export const THREDDS = "https://opendap.co-ops.nos.noaa.gov/thredds";
/** California box on the regulargrid (935×833, lat 18.42 + 0.04·ny, lon −145.38 + 0.04·nx): 32.38–42.10°N, 125.02–117.02°W. */
export const BOX = { ny0: 349, ny1: 592, nx0: 509, nx1: 709 } as const;
const LAT0 = 18.42, LON0 = -145.38, STEP = 0.04;
export const GEOMETRY = {
  width: BOX.nx1 - BOX.nx0 + 1,
  height: BOX.ny1 - BOX.ny0 + 1,
  lat_first: +(LAT0 + BOX.ny0 * STEP).toFixed(4),
  lat_step: STEP,
  lon_first: +(LON0 + BOX.nx0 * STEP).toFixed(4),
  lon_step: STEP,
};
export const FILL = -99999;
export const NEAR_SHORE_KM = 6;
/** 8 nowcast steps (n003…n024) and 24 forecast steps (f003…f072). */
export const STEP_IDS = [
  ...Array.from({ length: 8 }, (_, i) => `n${String((i + 1) * 3).padStart(3, "0")}`),
  ...Array.from({ length: 24 }, (_, i) => `f${String((i + 1) * 3).padStart(3, "0")}`),
];

export const SOURCE: OceanSource = {
  name: "NOAA CO-OPS West Coast Operational Forecast System (WCOFS)",
  product: "WCOFS regulargrid (0.04°, 3-hourly), depth 0 m",
  url: `${THREDDS}/catalog/NOAA/WCOFS/MODELS/catalog.html`,
  info_url: "https://tidesandcurrents.noaa.gov/ofs/wcofs/wcofs_info.html",
  licence: "U.S. Government work (NOAA), public domain",
  developmental: true,
};

export const CAVEATS = [
  "NOAA labels WCOFS developmental. Skill drops with lead time.",
  "A numerical model (ROMS with data assimilation), not a measurement: HF radar is the measured current, satellites the observed temperature.",
  "Nowcast steps are model analyses, not observations.",
  "About 4 km resolution: unsuitable inside bays, harbors and the surf zone; cells within about 6 km of shore are flagged as coarse near shore.",
  "Temperature is the model's 0 m level on an interpolated grid, not satellite skin temperature. Forecast river inflow uses climatology.",
  "WCOFS has no biology: it is not a bloom, toxin or chlorophyll prediction.",
];

/** Constraint brackets are percent-encoded: Tomcat (THREDDS) rejects raw brackets in a query. */
const BOXQ = `%5B0%5D%5B0%5D%5B${BOX.ny0}:1:${BOX.ny1}%5D%5B${BOX.nx0}:1:${BOX.nx1}%5D`;

const pad = (n: number) => String(n).padStart(2, "0");
const ymd = (d: Date) => `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}`;
const dir = (cycleId: string) => `NOAA/WCOFS/MODELS/${cycleId.slice(0, 4)}/${cycleId.slice(4, 6)}/${cycleId.slice(6, 8)}`;
const hh = (cycleId: string) => cycleId.slice(8, 10);

export const catalogUrl = (day: Date) => `${THREDDS}/catalog/NOAA/WCOFS/MODELS/${day.getUTCFullYear()}/${pad(day.getUTCMonth() + 1)}/${pad(day.getUTCDate())}/catalog.xml`;
export const stepUrl = (cycleId: string, step: string) =>
  `${THREDDS}/dodsC/${dir(cycleId)}/wcofs.t${hh(cycleId)}z.${cycleId.slice(0, 8)}.regulargrid.${step}.nc.dods?u_eastward${BOXQ},v_northward${BOXQ},temp${BOXQ},time`;

export function cycleTime(cycleId: string): Date {
  return new Date(Date.UTC(+cycleId.slice(0, 4), +cycleId.slice(4, 6) - 1, +cycleId.slice(6, 8), +hh(cycleId)));
}

/** Hours from the cycle time: nNNN is NNN hours into the 24 h nowcast that ends at the cycle; fNNN is NNN hours after. */
export function leadHours(step: string): number {
  const n = Number(step.slice(1));
  return step[0] === "n" ? n - 24 : n;
}

export function validTime(cycleId: string, step: string): string {
  return new Date(cycleTime(cycleId).getTime() + leadHours(step) * 3600e3).toISOString();
}

/** The newest cycle in a THREDDS day catalog and its regulargrid steps (in STEP_IDS order). */
export function parseCatalog(xml: string): { cycleId: string; steps: string[] } | null {
  const by = new Map<string, Set<string>>();
  for (const m of xml.matchAll(/name="wcofs\.t(\d\d)z\.(\d{8})\.regulargrid\.([nf]\d{3})\.nc"/g)) {
    const id = `${m[2]}${m[1]}`;
    if (!by.has(id)) by.set(id, new Set());
    by.get(id)!.add(m[3]);
  }
  const newest = [...by.keys()].sort().pop();
  if (!newest) return null;
  const have = by.get(newest)!;
  return { cycleId: newest, steps: STEP_IDS.filter((s) => have.has(s)) };
}

/** Probe today's catalog, then yesterday's and the day before (UTC). Throws only when every probe failed to answer. */
export async function discoverCycle(now: Date): Promise<{ cycleId: string; steps: string[] } | null> {
  let answered = 0;
  let lastErr = "";
  for (let back = 0; back < 3; back++) {
    const day = new Date(now.getTime() - back * 86400e3);
    let xml: string;
    try {
      xml = await getText(catalogUrl(day), { timeoutMs: 6000, revalidate: 3600 });
    } catch (e) {
      lastErr = errMsg(e);
      if (/HTTP 404/.test(lastErr)) answered++; // the day's folder does not exist yet
      continue;
    }
    answered++;
    const c = parseCatalog(xml);
    if (c && c.steps.length && c.cycleId.startsWith(ymd(day))) return c;
  }
  if (!answered) throw new Error(`THREDDS catalog unreachable (${lastErr})`);
  return null;
}

const familyInfo = (cycleId: string | null, step: WcofsStep | undefined) => {
  const label = cycleId && step ? wcofsLabel(cycleId, step) : "NOAA WCOFS model guidance (developmental) · unavailable";
  return {
    "currents.predicted": { family: "currents.predicted" as const, units: "m/s" as const, vars: ["u", "v"] as WcofsVar[], label },
    "sst.model": { family: "sst.model" as const, units: "°C" as const, vars: ["sst"] as WcofsVar[], label },
  };
};

export const gridPath = (cycleId: string, step: string, v: WcofsVar) => `/explore/model/wcofs/grid?cycle=${cycleId}&step=${step}&var=${v}`;

export function buildIndex(c: { cycleId: string; steps: string[] }, now: Date, fallback: string | null): WcofsIndex {
  const steps: WcofsStep[] = c.steps.map((id) => {
    const kind = id[0] === "n" ? ("nowcast" as const) : ("forecast" as const);
    return {
      id,
      kind,
      track: kind === "nowcast" ? ("estimates" as const) : ("forecasts" as const),
      lead_h: leadHours(id),
      valid_time: validTime(c.cycleId, id),
      grids: {
        u: gridRecord(GEOMETRY, PACK_VELOCITY, gridPath(c.cycleId, id, "u")),
        v: gridRecord(GEOMETRY, PACK_VELOCITY, gridPath(c.cycleId, id, "v")),
        sst: gridRecord(GEOMETRY, PACK_TEMPERATURE, gridPath(c.cycleId, id, "sst")),
        mask: gridRecord(GEOMETRY, PACK_MASK, gridPath(c.cycleId, id, "mask")),
      },
    };
  });
  const t = now.toISOString();
  const past = steps.filter((s) => s.valid_time <= t);
  const def = past.length ? past[past.length - 1] : steps[0];
  const valid = steps.length ? { start: steps[0].valid_time, end: steps[steps.length - 1].valid_time } : null;
  const expired = !!valid && valid.end < t;
  const reasons = [
    expired ? `The newest WCOFS cycle found (${cycleName(c.cycleId)}) is valid only until ${valid!.end}; no newer cycle was published.` : null,
    fallback ? `A newer cycle could not be checked (${fallback}); the last good cycle is shown.` : null,
  ].filter(Boolean);
  return {
    status: expired ? "expired" : "current",
    reason: reasons.length ? reasons.join(" ") : null,
    source: SOURCE,
    cycle: { id: c.cycleId, issued: cycleTime(c.cycleId).toISOString(), steps_available: steps.length, steps_expected: STEP_IDS.length },
    valid,
    fallback: !!fallback,
    default_step: def?.id ?? null,
    steps,
    families: familyInfo(c.cycleId, def),
    near_shore_km: NEAR_SHORE_KM,
    caveats: CAVEATS,
    generated_at: t,
  };
}

export function unavailableIndex(reason: string, now: Date): WcofsIndex {
  return {
    status: "unavailable", reason, source: SOURCE, cycle: null, valid: null, fallback: false, default_step: null, steps: [],
    families: familyInfo(null, undefined), near_shore_km: NEAR_SHORE_KM, caveats: CAVEATS, generated_at: now.toISOString(),
  };
}

let indexCache: { at: number; ttl: number; c: { cycleId: string; steps: string[] } | null; err: string | null } | null = null;
let lastGood: { cycleId: string; steps: string[] } | null = null;

/** The WCOFS index (cached; see the module comment). */
export async function wcofsIndex(): Promise<WcofsIndex> {
  const now = await oceanNow();
  const wall = Date.now();
  if (!indexCache || wall - indexCache.at > indexCache.ttl) {
    try {
      const c = await discoverCycle(now);
      const todays = c?.cycleId.startsWith(ymd(now));
      indexCache = { at: wall, ttl: todays ? 3600e3 : 900e3, c, err: null };
      if (c) lastGood = c;
    } catch (e) {
      indexCache = { at: wall, ttl: 300e3, c: null, err: errMsg(e) };
    }
  }
  if (indexCache.c) return buildIndex(indexCache.c, now, null);
  if (indexCache.err && lastGood) return buildIndex(lastGood, now, indexCache.err);
  return unavailableIndex(indexCache.err ? `WCOFS could not be read: ${indexCache.err}` : "No WCOFS cycle was published in the last three days.", now);
}

export type PackedStep = Record<WcofsVar, Buffer> & { valid_time: string };

/** Decode one step's `.dods` subset into packed, gzipped grids. Checks the file's own time against the step id. */
export function packStep(body: Uint8Array, cycleId: string, step: string): PackedStep {
  const vars = decodeDods(body);
  const u = vars.get("u_eastward"), v = vars.get("v_northward"), temp = vars.get("temp"), time = vars.get("time");
  if (!u || !v || !temp) throw new Error("WCOFS response is missing u_eastward, v_northward or temp");
  const n = GEOMETRY.width * GEOMETRY.height;
  if (u.data.length !== n || v.data.length !== n || temp.data.length !== n) throw new Error("WCOFS response has an unexpected grid size");
  const expected = validTime(cycleId, step);
  if (time) {
    const t = new Date(Date.UTC(2016, 0, 1) + Number(time.data[0]) * 1000).toISOString(); // seconds since 2016-01-01
    if (t !== expected) throw new Error(`WCOFS ${step} is valid at ${t}, expected ${expected}`);
  }
  const t = temp.data;
  const mask = shoreMask((k) => !isFill(t[k], FILL), GEOMETRY, NEAR_SHORE_KM);
  return {
    u: gzipCodes(packCodes(u.data, PACK_VELOCITY, FILL)),
    v: gzipCodes(packCodes(v.data, PACK_VELOCITY, FILL)),
    sst: gzipCodes(packCodes(t, PACK_TEMPERATURE, FILL)),
    mask: gzipCodes(mask),
    valid_time: expected,
  };
}

const steps = new PromiseLru<PackedStep>(48);

export const isCycleId = (s: string) => /^\d{8}(03|09|15|21)$/.test(s);
export const isStepId = (s: string) => STEP_IDS.includes(s);

/** One step's packed grids (fetched on first request, then cached). Cycles older than four days are refused. */
export async function wcofsStep(cycleId: string, step: string): Promise<PackedStep> {
  if (!isCycleId(cycleId) || !isStepId(step)) throw new RangeError("unknown cycle or step");
  const age = (await oceanNow()).getTime() - cycleTime(cycleId).getTime();
  if (age > 4 * 86400e3 || age < -6 * 3600e3) throw new RangeError("cycle outside the served window");
  return steps.get(`${cycleId}/${step}`, async () => packStep(await getBytes(stepUrl(cycleId, step), { timeoutMs: 10000, revalidate: 86400 }), cycleId, step));
}

/** Tests only. */
export function resetWcofsCaches() {
  indexCache = null;
  lastGood = null;
  steps.clear();
}
