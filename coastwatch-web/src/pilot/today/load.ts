/**
 * Server-side loading for Today at your port (/today). It reads only what the page needs: the
 * manifest, the ports, official notices, the C-HARM cells around the port, the derived outlook,
 * and the biotoxin and fisheries-status sidecars. It returns plain JSON inputs for `buildTodayModel`
 * (src/pilot/facts.ts). Ocean conditions are fetched by the page from /api/conditions (live NOAA,
 * one request), so the first HTML never waits on NWS.
 *
 * Every part fails on its own and says so. A missing sidecar is `null`, and the page shows that
 * section's unavailable state. Nothing is guessed.
 */
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { LayerArtifact, Manifest, PortsCollection, ValueGrid } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import { checkManifest, checkOfficial, checkPorts, dataBaseUrl } from "@/lib/data";
import { portPlaces, type CoastPlace } from "@/lib/coast";
import { associationsAt } from "@/lib/applicability";
import { CHARM_LEADS, charmLayer } from "@/lib/layers";
import { classifyTime } from "@/lib/freshness";
import { decodeCode, decodeGrid, cellAt, distanceKm } from "@/lib/grid";
import { neighbourhood, NEIGHBOURHOOD_KM } from "@/lib/hab";
import { derivedOutlook, parseDerived, type DerivedGrid } from "@/lib/derived";
import { PILOT_PORTS } from "@/pilot/contracts";
import type { CharmInput, CharmProb, NoticeInput, OutlookInput, PortTodayInputs } from "@/pilot/facts";
import { parseBiotoxin, parseFisheriesStatus } from "./sidecars";

export type PortOption = { place_id: string; label: string; county: string | null; pilot: boolean; lat: number; lon: number };

export type TodayLoad =
  | { ok: false; error: string }
  | { ok: true; place: CoastPlace | null; requested: string | null; ports: PortOption[]; inputs: Omit<PortTodayInputs, "conditions" | "selected"> | null; fixture: boolean };

export const fixedNow = () => (process.env.CW_FIXED_NOW ? new Date(process.env.CW_FIXED_NOW) : new Date());

const isHttp = (b: string) => /^https?:\/\//.test(b);
function localPath(base: string, rel: string) {
  const dir = process.env.CW_DATA_DIR || path.join(process.cwd(), "public", base);
  return path.join(dir, rel);
}
async function readJson(base: string, rel: string): Promise<unknown> {
  if (isHttp(base)) {
    const r = await fetch(`${base}/${rel}`, { next: { revalidate: 300 } });
    if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${rel}`);
    return r.json();
  }
  return JSON.parse(await readFile(localPath(base, rel), "utf8"));
}
const bytesCache = new Map<string, Promise<Uint16Array>>();
function readGrid(base: string, rel: string): Promise<Uint16Array> {
  const key = `${base}|${rel}`;
  let p = bytesCache.get(key);
  if (!p) {
    p = (async () => {
      let buf: ArrayBuffer;
      if (isHttp(base)) {
        const r = await fetch(`${base}/${rel}`, { next: { revalidate: 3600 } });
        if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${rel}`);
        buf = await r.arrayBuffer();
      } else {
        const b = await readFile(localPath(base, rel));
        buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
      }
      return decodeGrid(buf);
    })();
    p.catch(() => bytesCache.delete(key));
    // a small cache: the C-HARM grids and outlook chunks of the last few requests
    if (bytesCache.size > 64) bytesCache.delete(bytesCache.keys().next().value!);
    bytesCache.set(key, p);
  }
  return p;
}
async function optionalJson(base: string, rel: string | null | undefined): Promise<unknown | null> {
  if (!rel) return null;
  try {
    return await readJson(base, rel);
  } catch {
    return null;
  }
}

/** Pilot ports first (in the contract's order), then every other CDFW port from north to south. */
export function portOptions(ports: PortsCollection | null): PortOption[] {
  const all = portPlaces(ports);
  const pilot = PILOT_PORTS.map((p) => all.find((x) => x.key === p.place_id)).filter((x): x is CoastPlace => !!x);
  const rest = all.filter((x) => !PILOT_PORTS.some((p) => p.place_id === x.key)).sort((a, b) => b.lat - a.lat);
  const opt = (x: CoastPlace, pilotPort: boolean): PortOption => ({ place_id: x.key, label: PILOT_PORTS.find((p) => p.place_id === x.key)?.label ?? x.name, county: x.county, pilot: pilotPort, lat: x.lat, lon: x.lon });
  return [...pilot.map((x) => opt(x, true)), ...rest.map((x) => opt(x, false))];
}

function noticesAt(official: OfficialDataset | null, place: CoastPlace): NoticeInput[] | null {
  if (!official) return null;
  const verified = official.registry.review.status === "human_verified";
  return associationsAt(official, place.key, place.lat, place.lon)
    .filter((a) => a.tier === "geographic" && !a.dispute)
    .map((a) => ({
      id: a.record.id,
      agency: a.record.agency,
      title: a.record.title,
      summary: a.record.summary,
      action: a.record.action,
      species: [...a.record.species],
      fishery: a.record.fishery,
      effective: a.record.effective_date ?? null,
      url: a.record.sources[0]?.url ?? null,
      verified,
    }));
}

async function charmAt(base: string, m: Manifest, lat: number, lon: number, now: Date): Promise<CharmInput | null> {
  const lead0 = charmLayer(m, "particulate_domoic", 0) ?? charmLayer(m, "pseudo_nitzschia", 0);
  if (!lead0) return null;
  const read = async (l: LayerArtifact | null): Promise<CharmProb | null> => {
    if (!l?.grid) return null;
    try {
      const n = neighbourhood(l.grid as ValueGrid, await readGrid(base, l.grid.url), lat, lon);
      return { here: n.here, median: n.median, max: n.max, n: n.n, nearestKm: n.nearest?.km ?? null };
    } catch {
      return null;
    }
  };
  const days = await Promise.all(
    CHARM_LEADS.map(async (lead) => {
      const a = charmLayer(m, "particulate_domoic", lead);
      const b = charmLayer(m, "pseudo_nitzschia", lead);
      return { lead, date: (a ?? b)?.time.valid_date ?? null, pda: await read(a), pn: await read(b) };
    }),
  );
  const fresh = classifyTime(lead0.freshness, lead0.time, now);
  return { issued: lead0.time.issued_date ?? null, state: fresh.state, ageDays: fresh.ageDays ?? null, days, url: "https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/california/" };
}

/** The outlook in the port's own cell, else the median of its water cells within NEIGHBOURHOOD_KM (chunks read on the server). */
async function outlookAt(base: string, lat: number, lon: number): Promise<OutlookInput | null> {
  const raw = await optionalJson(base, "derived/index.json");
  const o = derivedOutlook(parseDerived(raw));
  if (!o) return null;
  const g: DerivedGrid = o.grid;
  let value: number | null = null;
  let basis: OutlookInput["basis"] = null;
  try {
    const c = cellAt(g, lat, lon);
    const dR = Math.ceil(NEIGHBOURHOOD_KM / 111 / Math.abs(g.lat_step)) + 1;
    const dC = Math.ceil(NEIGHBOURHOOD_KM / (111 * Math.cos((lat * Math.PI) / 180)) / Math.abs(g.lon_step)) + 1;
    const r0 = Math.floor((lat - g.lat_first) / g.lat_step + 0.5);
    const c0 = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
    const chunks = new Map<string, Uint16Array | null>();
    const codeAt = async (row: number, col: number): Promise<number | null> => {
      if (row < 0 || row >= g.height || col < 0 || col >= g.width) return null;
      const r = Math.floor(row / g.chunks.rows);
      const cc = Math.floor(col / g.chunks.cols);
      const key = `${r}_${cc}`;
      if (!g.chunks.present.includes(key)) return null;
      if (!chunks.has(key)) chunks.set(key, await readGrid(base, g.chunks.url_template.replace("{row}", String(r)).replace("{col}", String(cc))).catch(() => null));
      const codes = chunks.get(key);
      const width = Math.min(g.chunks.cols, g.width - cc * g.chunks.cols);
      return codes ? codes[(row - r * g.chunks.rows) * width + (col - cc * g.chunks.cols)] : null;
    };
    if (c) {
      const code = await codeAt(c.row, c.col);
      const v = code == null ? null : decodeCode(g, code);
      if (v != null) {
        value = v;
        basis = "cell";
      }
    }
    if (value == null) {
      const vals: number[] = [];
      for (let r = r0 - dR; r <= r0 + dR; r++)
        for (let cc = c0 - dC; cc <= c0 + dC; cc++) {
          const code = await codeAt(r, cc);
          const v = code == null ? null : decodeCode(g, code);
          if (v == null) continue;
          if (distanceKm(lat, lon, g.lat_first + r * g.lat_step, g.lon_first + cc * g.lon_step) <= NEIGHBOURHOOD_KM) vals.push(v);
        }
      vals.sort((a, b) => a - b);
      if (vals.length) {
        const mid = (vals.length - 1) / 2;
        value = (vals[Math.floor(mid)] + vals[Math.ceil(mid)]) / 2;
        basis = "nearby";
      }
    }
  } catch {
    value = null;
  }
  return { id: o.id, label: o.label, issued: o.issued, valid_from: o.valid_from, valid_to: o.valid_to, value, range: null, basis, skill: o.skill as OutlookInput["skill"], caveats: o.caveats, method: o.method };
}

export async function loadToday(placeKey: string | null): Promise<TodayLoad> {
  const base = dataBaseUrl();
  let manifest: Manifest;
  try {
    const m = checkManifest(await readJson(base, "manifest.json"));
    if (!m.ok) return { ok: false, error: m.error };
    manifest = m.manifest;
  } catch (e) {
    return { ok: false, error: `manifest not available (${(e as Error).message})` };
  }
  const pr = await optionalJson(base, manifest.ports_url);
  const pc = pr ? checkPorts(pr) : null;
  const ports = pc?.ok ? pc.ports : null;
  const options = portOptions(ports);
  const fixture = manifest.pipeline_version === "fixture" || manifest.pipeline_run_id === "fixture";
  const place = placeKey ? (portPlaces(ports).find((p) => p.key === placeKey) ?? null) : null;
  if (!place) return { ok: true, place: null, requested: placeKey, ports: options, inputs: null, fixture };
  const label = PILOT_PORTS.find((p) => p.place_id === place.key)?.label ?? place.name;
  const now = fixedNow();
  const [officialRaw, charm, outlook, bio, fish] = await Promise.all([
    optionalJson(base, manifest.official_url),
    charmAt(base, manifest, place.lat, place.lon, now).catch(() => null),
    outlookAt(base, place.lat, place.lon).catch(() => null),
    optionalJson(base, "biotoxin/index.json"),
    optionalJson(base, "fisheries-status/index.json"),
  ]);
  const oc = officialRaw ? checkOfficial(officialRaw) : null;
  const official = oc?.ok ? oc.official : null;
  const biotoxin = parseBiotoxin(bio);
  const fisheries = parseFisheriesStatus(fish);
  return {
    ok: true,
    place,
    requested: placeKey,
    ports: options,
    fixture,
    inputs: {
      place: { place_id: place.key, label, lat: place.lat, lon: place.lon },
      now: now.toISOString(),
      dataGeneratedAt: manifest.generated_at,
      fisheries,
      // only this port's sites travel to the page
      biotoxin: biotoxin ? { ...biotoxin, sites: biotoxin.sites.filter((s) => s.informs_places.includes(place.key)) } : null,
      notices: noticesAt(official, place),
      charm,
      outlook,
    },
  };
}
