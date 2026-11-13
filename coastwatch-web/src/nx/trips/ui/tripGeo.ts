import type { Bounds } from "@/nx/chart/mercator";
import { RAMP_FISHERIES, RAMP_ZONES, rampZoneAt, shoreLon, type RampZone } from "@/nx/chart/regulation";
import type { EvidenceItem, Snapshot, Trip } from "../types";

/**
 * What a trip's chart draws, read from the trip and its snapshot (pure; the unit test reads it): the port, the
 * RAMP zone it lies in (for the crab fisheries RAMP governs), the CDPH test site and the buoy where the snapshot
 * names their position, the NWS zone and its advisories. Nothing is placed that the record gives no position for:
 * a test without a known site is listed, never drawn at a guessed place. No route, distance or ground is drawn.
 */
export type GeoPort = { key: string; name: string; lon: number; lat: number };

export type GeoTest = { key: string; site: string; lon: number; lat: number; date: string | null; max: number | null; action: number | null; old: boolean; url: string | null };
export type GeoBuoy = { key: string; id: string; lon: number; lat: number; at: string | null; waveFt: number | null; periodS: number | null; windKt: number | null; windDir: string | null; url: string | null };

export type TripGeo = {
  port: GeoPort;
  zone: RampZone | null;
  /** the whale-zone record's own words ("Fleet Advisory"), with its link; null when the snapshot holds none */
  whale: { text: string; url: string | null; verified: boolean | null } | null;
  test: GeoTest | null;
  buoy: GeoBuoy | null;
  nwsZone: string | null;
  advisories: string[];
  /** the box the chart fits: the port, its zone's nearer limit at the shore, the buoy and the test site, sea to the west */
  bounds: Bounds;
  /** when the snapshot was taken, and whether it was a live read or a saved copy */
  takenAt: string | null;
  mode: Snapshot["mode"] | null;
};

type Fields = EvidenceItem["fields"];

/** The first of several field names that holds a finite number (the snapshot's field names are the source's own). */
export function num(f: Fields | undefined, ...names: string[]): number | null {
  if (!f) return null;
  for (const n of names) {
    const v = f[n];
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
  }
  return null;
}

export function str(f: Fields | undefined, ...names: string[]): string | null {
  if (!f) return null;
  for (const n of names) {
    const v = f[n];
    if (typeof v === "string" && v.trim()) return v.trim();
    if (typeof v === "number" && Number.isFinite(v)) return String(v);
  }
  return null;
}

const LAT = ["lat", "latitude"];
const LON = ["lon", "lng", "longitude"];

/** A plausible California-coast position: a record with no position is not drawn. */
const placed = (lat: number | null, lon: number | null, port: GeoPort): lat is number =>
  lat != null && lon != null && Math.abs(lat - port.lat) < 1.2 && Math.abs(lon - port.lon) < 1.5;

export function tripGeo(trip: Pick<Trip, "fishery" | "placeKey" | "departKey">, snapshot: Snapshot | null, port: GeoPort): TripGeo {
  const ev = snapshot?.evidence ?? [];
  const ramp = RAMP_FISHERIES.has(trip.fishery);
  const whaleItem = ev.find((e) => e.kind === "whale") ?? null;
  const zoneId = ramp ? (str(whaleItem?.fields, "zone") ?? rampZoneAt(port.lat)) : null;
  const zone = RAMP_ZONES.find((z) => z.zone === zoneId) ?? null;

  const t = ev.find((e) => e.kind === "cdphTest") ?? null;
  const tLat = num(t?.fields, ...LAT);
  const tLon = num(t?.fields, ...LON);
  const test: GeoTest | null =
    t && placed(tLat, tLon, port)
      ? {
          key: t.key,
          // the site's own name: a field when the snapshot carries one, else the last part of "CDPH test, <species>, <site>"
          site: str(t.fields, "site", "siteLabel") ?? t.title.split(",").pop()!.trim(),
          lat: tLat,
          lon: tLon!,
          date: str(t.fields, "date", "collected") ?? t.asOf,
          max: num(t.fields, "max", "maxPpm", "max_ppm"),
          action: num(t.fields, "action", "actionPpm", "action_ppm"),
          old: t.stale || (num(t.fields, "ageDays") ?? 0) > 30,
          url: t.url,
        }
      : null;

  // the wave buoy first (it is the one the sea window reads); a wind-only station when it is the only one placed
  const obs = ev.filter((e) => e.kind === "observation" && e.source === "NDBC");
  const b = obs.find((e) => placed(num(e.fields, ...LAT), num(e.fields, ...LON), port) && num(e.fields, "waveFt") != null) ?? obs.find((e) => placed(num(e.fields, ...LAT), num(e.fields, ...LON), port)) ?? null;
  const buoy: GeoBuoy | null = b
    ? {
        key: b.key,
        id: str(b.fields, "id", "station") ?? b.key.replace(/^obs:/, ""),
        lat: num(b.fields, ...LAT)!,
        lon: num(b.fields, ...LON)!,
        at: b.asOf,
        waveFt: num(b.fields, "waveFt"),
        periodS: num(b.fields, "periodS"),
        windKt: num(b.fields, "windKt"),
        windDir: str(b.fields, "windDir"),
        url: b.url,
      }
    : null;

  const advisories = ev.filter((e) => e.kind === "advisory").map((e) => str(e.fields, "event") ?? e.title);
  return {
    port,
    zone,
    whale: whaleItem ? { text: str(whaleItem.fields, "measures") ?? whaleItem.text, url: whaleItem.url, verified: whaleItem.verified } : null,
    test,
    buoy,
    nwsZone: snapshot?.forecast?.zone ?? str(ev.find((e) => e.source === "NWS")?.fields, "zone") ?? null,
    advisories: [...new Set(advisories)],
    bounds: geoBounds(port, zone, [buoy ? [buoy.lon, buoy.lat] : null, test ? [test.lon, test.lat] : null]),
    // a copy built from saved reads carries the time it was read, never the time it was assembled
    takenAt: snapshot ? (snapshot.mode === "saved" ? (snapshot.savedFrom ?? snapshot.takenAt) : snapshot.takenAt) : null,
    mode: snapshot?.mode ?? null,
  };
}

type LngLat = [number, number];

/**
 * The box: the port, the water south of it, the nearer limit of its zone where it meets the shore (when within
 * about 80 km), the buoy and the test site; at least about 40 km tall, with half as much again of sea to the west.
 * The same framing as My Coast's trip view (src/nx/coast/CoastMap.tsx tripScene).
 */
export function geoBounds(port: GeoPort, zone: RampZone | null, extra: Array<LngLat | null>): Bounds {
  const pts: LngLat[] = [
    [port.lon, port.lat],
    [port.lon, port.lat - 0.16],
  ];
  for (const p of extra) if (p && Math.abs(p[1] - port.lat) < 0.8 && Math.abs(p[0] - port.lon) < 1) pts.push(p);
  if (zone) {
    const lat = port.lat - zone.south < zone.north - port.lat ? zone.south : zone.north;
    const d = Math.abs(lat - port.lat);
    if (d < 0.75) pts.push([shoreLon(lat) ?? port.lon, lat]);
  }
  let w = Math.min(...pts.map((p) => p[0]));
  let e = Math.max(...pts.map((p) => p[0]));
  let s = Math.min(...pts.map((p) => p[1]));
  let n = Math.max(...pts.map((p) => p[1]));
  const minLat = 0.36;
  if (n - s < minLat) {
    const c = (n + s) / 2;
    s = c - minLat / 2;
    n = c + minLat / 2;
  }
  const pad = (n - s) * 0.1;
  w -= Math.max(0.18, (e - w) * 0.5);
  e += 0.06;
  return [
    [w, s - pad],
    [e, n + pad],
  ];
}
