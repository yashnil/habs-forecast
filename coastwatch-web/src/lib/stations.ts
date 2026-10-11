import type { StationLite } from "@/lib/data";
import { distanceKm } from "@/lib/grid";

/** Stations within this distance are "nearby": about the reach of one bay or stretch of coast. */
export const NEARBY_KM = 30;

/** Monitoring stations near a point, nearest first; stations whose last update failed are left out. */
export function nearbyStations(stations: StationLite[], lat: number, lon: number, max = 2) {
  return stations
    .filter((s) => s.status !== "failed")
    .map((s) => ({ s, km: distanceKm(lat, lon, s.lat, s.lon) }))
    .filter((x) => x.km <= NEARBY_KM)
    .sort((a, b) => a.km - b.km)
    .slice(0, max);
}
