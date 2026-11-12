"use client";

/** The scenes the shell and pages share: the whole coast, the opening's cover, a port's water. */
import { boundsAround, type ChartScene as Scene, type SceneView } from "@/nx/chart/scene";
import { RAMP_ZONES, eastmostShore, shoreLon } from "@/nx/chart/regulation";
import { STILL } from "@/nx/chart/style";
import type { Bounds } from "@/nx/chart/mercator";
import type { ShellPort } from "./port";

/** California's coast, Oregon to Mexico: the view with no port chosen. */
export const CA_COAST: Bounds = [
  [-124.9, 32.4],
  [-117.0, 42.0],
];

/**
 * The view with no port: the coast against the sheet's east side and its sea west of it, out past the shelf, so
 * the spare width is water rather than Nevada (the box itself runs a little offshore of Point Arena).
 */
export const COAST_VIEW: SceneView = {
  bounds: [
    [-126.6, 32.4],
    [-117.0, 42.0],
  ],
  align: [1, 0.5],
};

export const HOME_SCENE: Scene = { id: "home", view: { cover: STILL.bounds, position: [0.5, 0.5] }, interactive: false };

const focusOf = (p: ShellPort | null) => (p ? { key: p.key, name: p.name, lon: p.lon, lat: p.lat } : null);

/** A destination's scene around the current port, or the whole coast. `km` is the half-width around the port. */
export function portScene(page: string, port: ShellPort | null, km: number): Scene {
  if (!port) return { id: `${page}:coast`, view: COAST_VIEW };
  // California's water is to the west: the view leans seaward, so the port sits right of centre with its sea before it
  const lean = (km * 0.45) / (111.32 * Math.cos((port.lat * Math.PI) / 180));
  return { id: `${page}:${port.key}`, view: { bounds: boundsAround([port.lon - lean, port.lat], km, km * 0.8) }, focus: focusOf(port) };
}

/**
 * A port's RAMP zone in one view (My Coast for a crab fishery): both boundary lines, the port, and the sea out
 * past the shelf, so the regulation that gates the trip is drawn whole. Null when the port is in no zone.
 */
export function zoneScene(page: string, port: ShellPort, zone: string | null): Scene | null {
  const z = RAMP_ZONES.find((x) => x.zone === zone);
  if (!z) return null;
  const s = z.south - 0.06, n = z.north + 0.06;
  const west = Math.min(shoreLon(z.north) ?? port.lon, shoreLon(z.south) ?? port.lon, port.lon) - 0.45;
  const east = Math.max(eastmostShore(z.south, z.north) ?? port.lon, port.lon) + 0.12;
  return {
    id: `${page}:${port.key}:zone-${z.zone}`,
    view: { bounds: [[west, s], [east, n]], align: [1, 0.5] },
    focus: focusOf(port),
  };
}
