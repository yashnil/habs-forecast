import { useMemo } from "react";
import { useGrid, useManifest, useObservations, useOfficial, usePorts } from "../data/store";
import type { LngLat } from "../lib/geo";
import { modelReading, nearestPort, noticeGeometry, type Reading, stationReading } from "../lib/provenance";
import { forecastFreshness } from "../lib/time";

/** The one state every direction renders: a point off Santa Cruz, C-HARM particulate domoic acid
 *  for the lead whose valid day is "today" on the fixed clock (lead 2), computed from real data. */
export const POINT: LngLat = [-122.06, 36.8];
export const LEAD = 2;
export const VARIABLE = "particulate_domoic";

export function useDirectionState() {
  const manifest = useManifest();
  const m = manifest.data;
  const official = useOfficial(m);
  const observations = useObservations(m);
  const ports = usePorts(m);
  const layer = m?.layers.find((l) => l.layer_id === `charm_${VARIABLE}_lead${LEAD}`);
  const grid = useGrid(layer);
  const run = m?.forecast_runs.find((r) => r.group_id === "charm");
  const charmSrc = m?.sources.find((s) => s.source_id === "charm");
  const freshness = run ? forecastFreshness(run.issued_date, charmSrc) : null;
  const reading: Reading | null = useMemo(() => {
    if (!official.data || !observations.data || !ports.data) return null;
    const np = nearestPort(ports.data, POINT);
    return {
      point: POINT,
      place: { label: np ? `Off ${np.port.display_name}` : "Open water", port: np?.port, portKm: np?.km },
      model: modelReading(grid.data, layer, POINT, m, grid.status === "failed"),
      satellite: { status: "off" },
      station: stationReading(observations.data, POINT),
      ...noticeGeometry(official.data, POINT),
    };
  }, [official.data, observations.data, ports.data, grid.data, grid.status, layer, m]);
  const covering = reading ? reading.notices.filter((n) => n.tier === "geographic") : [];
  const limitNotice = reading?.notices.filter((n) => n.nearestLimit && n.tier === "geographic").sort((a, b) => a.nearestLimit!.km - b.nearestLimit!.km)[0] ?? null;
  const limit = limitNotice?.nearestLimit ?? null;
  return { m, layer, grid: grid.data, official: official.data, observations: observations.data, ports: ports.data, reading, covering, limit, limitNotice, freshness, run, ready: !!reading && reading.model.status !== "loading" };
}
export type DirState = ReturnType<typeof useDirectionState>;

export const fmtLat = (lat: number) => `${Math.floor(lat)}°${((lat - Math.floor(lat)) * 60).toFixed(2)}′ N`;
export const shortAction: Record<string, string> = { quarantine: "Quarantine", consumption_advisory: "Health advisory", take_restriction: "Take restriction", fishery_closure: "Fishery closure", special_advisory: "Special advisory" };
export function classRange(v: number) {
  const c = Math.min(9, Math.floor(v * 10 + 1e-9));
  return { c, lo: c * 10, hi: c * 10 + 10 };
}
