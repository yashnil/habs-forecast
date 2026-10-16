import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { BUOYS, TIDE_STATIONS, WATER_TEMP_STATIONS } from "@/lib/noaaStations";
import {
  harborCandidate,
  nearest,
  parseAlerts,
  parseNdbc,
  parseTides,
  parseWaterTempSeries,
  parseZoneForecast,
  stationName,
  HARBOR_TEMP_MAX_AGE_H,
  HARBOR_TEMP_MAX_KM,
  NWS_MARINE_PAGE,
  type Conditions,
  type Part,
  type WaterTempSeries,
} from "@/lib/conditions";
import { parseShoreCsv, shoreCandidate, shorePage, shoreUrl, SHORE_MAX_AGE_H, SHORE_MAX_KM } from "@/lib/shoreStations";
import { headlineAlerts, parseCoastalText, zoneNear, zonePageUrl, zoneTextUrl } from "@/lib/nwsText";

/**
 * Ocean conditions for one place, from NWS, NDBC and CO-OPS (see lib/conditions.ts).
 * Each part fails on its own and says why; nothing is invented or carried forward.
 * Responses are cached for 10 minutes; each upstream request has a 6 s budget.
 *
 * Tests set CW_CONDITIONS_FIXTURE to a JSON file of recorded responses (keyed by the sha1 of the
 * URL) instead of the network, and CW_FIXED_NOW for the clock. CW_CONDITIONS_RECORD_DIR records
 * responses one file per URL; tests/conditions-fixtures.json is those files merged.
 *
 * NWS: api.weather.gov's robots.txt disallows every automated client, so its marine forecast and
 * alerts are fetched only where an operator has set CW_NWS_API=1 (tests set it, with recorded
 * responses). Otherwise, for waters near a place Neritic knows, the zone's forecast and headlines
 * are read from NWS's plain-text product on tgftp.nws.noaa.gov (no robots.txt; lib/nwsText.ts),
 * unless CW_NWS_TEXT=0; elsewhere the card links to NWS's own marine forecast pages. NDBC and
 * CO-OPS, whose rules allow it, are fetched as before.
 */
export const dynamic = "force-dynamic";

const UA = "Neritic (formerly CoastWatch; California coastal information research project; https://github.com/yashnil/habs-forecast)";
const NWS = "https://api.weather.gov";

/** `head`: fetch only the first bytes (NDBC files list the newest rows first; ~8 KB ≈ the last 16 h). */
let recorded: Promise<Record<string, string>> | null = null;

async function get(url: string, as: "json" | "text", head?: number): Promise<unknown> {
  const fx = process.env.CW_CONDITIONS_FIXTURE;
  if (fx) {
    recorded ??= readFile(fx, "utf8").then((t) => JSON.parse(t) as Record<string, string>);
    const body = (await recorded)[createHash("sha1").update(url).digest("hex") + (as === "json" ? ".json" : ".txt")];
    if (body == null) throw new Error("HTTP 503 (not recorded)"); // a missing recording fails like an outage
    return as === "json" ? JSON.parse(body) : body;
  }
  const r = await fetch(url, {
    headers: { "User-Agent": UA, Accept: as === "json" ? "application/geo+json, application/json" : "text/plain", ...(head ? { Range: `bytes=0-${head - 1}` } : {}) },
    signal: AbortSignal.timeout(6000),
    next: { revalidate: 600 },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  let body = await r.text();
  // a ranged read ends mid-line: keep whole lines only
  if (head && r.status === 206) body = body.slice(0, body.lastIndexOf("\n") + 1);
  const rec = process.env.CW_CONDITIONS_RECORD_DIR; // recording test fixtures (development only; zone outlines are not used and not kept)
  if (rec) {
    await mkdir(rec, { recursive: true });
    const keep = as === "json" ? JSON.stringify({ ...JSON.parse(body), geometry: null }) : body;
    await writeFile(path.join(rec, createHash("sha1").update(url).digest("hex") + (as === "json" ? ".json" : ".txt")), keep);
  }
  return as === "json" ? JSON.parse(body) : body;
}

const now = () => (process.env.CW_FIXED_NOW ? new Date(process.env.CW_FIXED_NOW) : new Date());
const msg = (e: unknown) => (e instanceof Error ? e.message : String(e)).slice(0, 120);

/**
 * Coastal marine zones for these waters, nearest first. A port or harvest site is on land, so the
 * nearest water is searched in rings (about 2, 5 and 10 km, eight directions); stepping only west
 * would cross headlands such as the Monterey peninsula into the wrong zone.
 */
async function zonesFor(lat: number, lon: number): Promise<{ id: string; name: string }[]> {
  const at = async (la: number, lo: number) => {
    const j = (await get(`${NWS}/zones?type=coastal&point=${la.toFixed(4)},${lo.toFixed(4)}&include_geometry=false`, "json")) as { features?: { properties: { id: string; name: string } }[] };
    const f = j.features?.[0];
    return f ? { id: f.properties.id, name: f.properties.name } : null;
  };
  const out: { id: string; name: string }[] = [];
  const add = (z: { id: string; name: string } | null) => z && !out.some((x) => x.id === z.id) && out.push(z);
  add(await at(lat, lon));
  if (out.length) return out;
  const kx = 1 / (111.32 * Math.cos((lat * Math.PI) / 180));
  for (const km of [2, 5, 10]) {
    const ring = await Promise.all(
      [0, 45, 90, 135, 180, 225, 270, 315].map((deg) => {
        const r = (deg * Math.PI) / 180;
        return at(lat + (km * Math.cos(r)) / 111.32, lon + km * Math.sin(r) * kx).catch(() => null);
      }),
    );
    ring.forEach(add); // nearest ring first; farther rings add fallbacks (e.g. past a harbor-bar zone with no forecast)
  }
  return out;
}

async function marineParts(lat: number, lon: number): Promise<[Conditions["marine"], Conditions["alerts"]]> {
  const source = "NOAA National Weather Service";
  if (process.env.CW_NWS_API !== "1") {
    const link: Part<never> = { status: "none", reason: "Read the marine forecast and any warnings for your waters on NWS's site.", source, url: NWS_MARINE_PAGE };
    const z = process.env.CW_NWS_TEXT === "0" ? null : zoneNear(lat, lon);
    if (!z) return [link, link];
    try {
      const r = parseCoastalText(z.zone, (await get(zoneTextUrl(z.zone), "text")) as string);
      if (!r) return [{ ...link, status: "unavailable", reason: `NWS's text product did not contain zone ${z.zone}.` }, link];
      const fetchedAt = now().toISOString();
      return [
        { status: "ok", data: r.forecast, source, url: zonePageUrl(z.zone), fetchedAt },
        { status: "ok", data: headlineAlerts(r.headlines, zonePageUrl(z.zone)), source, url: zonePageUrl(z.zone), fetchedAt },
      ];
    } catch (e) {
      return [{ status: "unavailable", reason: `The NWS forecast did not load (${msg(e)}).`, source, url: zonePageUrl(z.zone) }, link];
    }
  }
  let zones: { id: string; name: string }[];
  try {
    zones = await zonesFor(lat, lon);
  } catch (e) {
    const u: Part<never> = { status: "unavailable", reason: `The NWS zone service did not answer (${msg(e)}).`, source };
    return [u, u];
  }
  if (!zones.length) {
    const n: Part<never> = { status: "none", reason: "These waters are not inside an NWS coastal marine zone.", source };
    return [n, n];
  }
  // some zones (harbor bars) have no zone forecast: use the first nearby zone that has one
  let lastError = "";
  for (const zone of zones) {
    const furl = `${NWS}/zones/coastal/${zone.id}/forecast`;
    const aurl = `${NWS}/alerts/active?zone=${zone.id}`;
    const [f, a] = await Promise.allSettled([get(furl, "json"), get(aurl, "json")]);
    const forecast = f.status === "fulfilled" ? parseZoneForecast(zone.id, zone.name, f.value) : null;
    if (!forecast) {
      lastError = f.status === "rejected" ? `The NWS forecast did not load (${msg(f.reason)}).` : "NWS returned no forecast periods.";
      continue;
    }
    const fetchedAt = now().toISOString();
    return [
      { status: "ok", data: forecast, source, url: `https://marine.weather.gov/MapClick.php?zoneid=${zone.id}`, fetchedAt },
      a.status === "fulfilled"
        ? { status: "ok", data: parseAlerts(a.value), source, url: `https://forecast.weather.gov/shmrn.php?mz=${zone.id.toLowerCase()}`, fetchedAt }
        : { status: "unavailable", reason: `NWS alerts did not load (${msg(a.reason)}).`, source, url: aurl },
    ];
  }
  const u: Part<never> = { status: "unavailable", reason: lastError || "No NWS forecast for these waters.", source };
  return [u, u];
}

async function buoyPart(lat: number, lon: number): Promise<Conditions["buoy"]> {
  const source = "NOAA National Data Buoy Center";
  const cands = nearest(BUOYS, lat, lon, 8, 60);
  if (!cands.length) return { status: "none", reason: "No NDBC station reports within 60 km.", source };
  const errors: string[] = [];
  for (const s of cands) {
    const url = `https://www.ndbc.noaa.gov/data/realtime2/${s.id}.txt`;
    try {
      const obs = parseNdbc((await get(url, "text", 8192)) as string, now());
      if (Object.keys(obs).length)
        return { status: "ok", data: obs, source, url: `https://www.ndbc.noaa.gov/station_page.php?station=${s.id}`, fetchedAt: now().toISOString(), station: { id: s.id, name: s.name, km: s.km, lat: s.lat, lon: s.lon } };
      errors.push(`${s.id}: nothing in the last 6 h`);
    } catch (e) {
      errors.push(`${s.id}: ${msg(e)}`);
    }
  }
  return { status: "unavailable", reason: `No recent observation from the nearest stations (${errors.join("; ")}).`, source };
}

async function tidePart(lat: number, lon: number): Promise<Conditions["tides"]> {
  const source = "NOAA CO-OPS tide predictions";
  const cands = nearest(TIDE_STATIONS, lat, lon, 3, 40);
  if (!cands.length) return { status: "none", reason: "No tide prediction station within 40 km.", source };
  const d = now();
  // predictions are requested in local time from the start of today in California
  const ymd = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit" }).format(d).replaceAll("-", "");
  const errors: string[] = [];
  // some stations publish no high/low predictions: use the nearest that does
  for (const s of cands) {
    const url = `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions&station=${s.id}&datum=MLLW&interval=hilo&units=english&time_zone=lst_ldt&format=json&begin_date=${ymd}&range=96&application=CoastWatch`;
    try {
      const tides = parseTides(await get(url, "json"));
      if (tides.length)
        return { status: "ok", data: tides, source, url: `https://tidesandcurrents.noaa.gov/noaatidepredictions.html?id=${s.id}`, fetchedAt: d.toISOString(), station: { id: s.id, name: s.name, km: s.km, lat: s.lat, lon: s.lon } };
      errors.push(`${s.id}: no predictions`);
    } catch (e) {
      errors.push(`${s.id}: ${msg(e)}`);
    }
  }
  return { status: "unavailable", reason: `No tide predictions from the nearest stations (${errors.join("; ")}).`, source };
}

/** "20261013 03:00" in GMT, the form CO-OPS takes for begin_date/end_date. */
const coopsStamp = (d: Date) => d.toISOString().slice(0, 16).replace("-", "").replace("-", "").replace("T", " ");

/**
 * Water temperature in the harbour: CO-OPS's 6-minute series at the single nearest tide station with a
 * thermistor, and only if it is within HARBOR_TEMP_MAX_KM (6 km, harbour/bay scale). Farther, the part is
 * "none": a thermistor across the bay (Alameda for San Francisco) is not this harbour's water, so there
 * is no fallback to the next station. The window is the last HARBOR_TEMP_MAX_AGE_H hours up to now, so
 * the request is reproducible for a fixed clock. A station whose sensor is down answers `{"error":
 * "No data was found…"}`, reported as "sensor not reporting", not as an outage of the service; an HTTP
 * error (or, in tests, a missing recording) is the outage. `basis` says whether the station is in this
 * harbour (≤ HARBOR_TEMP_AT_KM) or a nearby one named with its distance (docs/coastwatch/v2/03-coops-audit.md).
 */
async function harborPart(lat: number, lon: number): Promise<Conditions["harbor"]> {
  const source = "NOAA CO-OPS water temperature";
  const s = harborCandidate(WATER_TEMP_STATIONS, lat, lon);
  if (!s) return { status: "none", reason: `No NOAA water-temperature station within ${HARBOR_TEMP_MAX_KM} km.`, source };
  const d = now();
  const begin = encodeURIComponent(coopsStamp(new Date(d.getTime() - HARBOR_TEMP_MAX_AGE_H * 3600_000)));
  const end = encodeURIComponent(coopsStamp(d));
  const url = `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_temperature&station=${s.id}&begin_date=${begin}&end_date=${end}&units=metric&time_zone=gmt&format=json&application=neritic`;
  const page = `https://tidesandcurrents.noaa.gov/stationhome.html?id=${s.id}`;
  const where = `NOAA tide station ${s.id} (${stationName(s.name)}, ${s.km.toFixed(1)} km)`;
  const station = { id: s.id, name: s.name, km: s.km, lat: s.lat, lon: s.lon, ...(s.sensorNote ? { sensorNote: s.sensorNote } : {}) };
  let series: WaterTempSeries;
  try {
    series = parseWaterTempSeries(await get(url, "json"), d);
  } catch (e) {
    return { status: "unavailable", reason: `${where}: the CO-OPS service did not answer (${msg(e)}).`, source, url: page };
  }
  if (series.reading) return { status: "ok", data: series.reading, source, url: page, fetchedAt: d.toISOString(), station, basis: s.basis };
  const why = series.error
    ? "sensor not reporting"
    : series.rows === 0
      ? `no readings in the last ${HARBOR_TEMP_MAX_AGE_H} h`
      : series.flagged && !series.outOfWindow
        ? `every reading in the last ${HARBOR_TEMP_MAX_AGE_H} h failed NOAA's quality checks`
        : `no reading in the last ${HARBOR_TEMP_MAX_AGE_H} h passed NOAA's checks (${series.flagged} flagged, ${series.outOfWindow} outside the window)`;
  return { status: "unavailable", reason: `${where}: ${why}.`, source, url: page };
}

/**
 * A CeNCOOS shore station's newest row (lib/shoreStations), when one is within SHORE_MAX_KM. Values
 * whose QARTOD flag is not PASS are withheld and named; a newest row older than SHORE_MAX_AGE_H
 * hours is "not reporting since …", said with the last report's time, never shown as current.
 */
async function shorePart(lat: number, lon: number): Promise<Conditions["shore"]> {
  const source = "CeNCOOS shore station";
  const s = shoreCandidate(lat, lon);
  if (!s) return { status: "none", reason: `No CeNCOOS shore station within ${SHORE_MAX_KM} km.`, source };
  const d = now();
  const page = shorePage(s);
  const station = { id: s.id, name: s.name, km: s.km, lat: s.lat, lon: s.lon, sensorNote: s.place };
  let obs;
  try {
    obs = parseShoreCsv((await get(shoreUrl(s, d), "text")) as string);
  } catch (e) {
    return { status: "unavailable", reason: `${s.name}: CeNCOOS ERDDAP did not answer (${msg(e)}).`, source, url: page };
  }
  if (!obs) return { status: "unavailable", reason: `${s.name}: no rows.`, source, url: page };
  const ageH = (d.getTime() - Date.parse(obs.at)) / 3600_000;
  if (ageH > SHORE_MAX_AGE_H) return { status: "unavailable", reason: `${s.name} (${s.km.toFixed(1)} km) has not reported since ${new Date(obs.at).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric", timeZone: "UTC" })}.`, source, url: page };
  return { status: "ok", data: obs, source, url: page, fetchedAt: d.toISOString(), station };
}

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const lat = Number(q.get("lat"));
  const lon = Number(q.get("lon"));
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < 32 || lat > 42.5 || lon < -127 || lon > -116) {
    return NextResponse.json({ error: "lat and lon must be California coastal coordinates" }, { status: 400 });
  }
  const [[marine, alerts], buoy, tides, harbor, shore] = await Promise.all([marineParts(lat, lon), buoyPart(lat, lon), tidePart(lat, lon), harborPart(lat, lon), shorePart(lat, lon)]);
  const body: Conditions = { lat, lon, marine, alerts, buoy, tides, harbor, shore };
  return NextResponse.json(body, { headers: { "Cache-Control": "public, s-maxage=600, stale-while-revalidate=300" } });
}
