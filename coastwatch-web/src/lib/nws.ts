/**
 * NWS Marine MapClick — point-based forecast pages (official).
 * Centers are approximate for each coastal band; users still verify zone text on the NWS page.
 */
const REGION_CENTERS: Record<string, { lat: number; lon: number; label: string }> = {
  north_coast: { lat: 41.2, lon: -124.2, label: "North CA offshore" },
  north_bay: { lat: 38.85, lon: -123.6, label: "Mendocino / Sonoma coast" },
  central_bay: { lat: 37.85, lon: -122.55, label: "SF Bay / Marin coast" },
  monterey: { lat: 36.75, lon: -122.05, label: "Monterey Bay area" },
  morro: { lat: 35.35, lon: -120.9, label: "Morro / Big Sur coast" },
  southern: { lat: 33.75, lon: -118.35, label: "Southern CA coast" },
};

export function marineForecastUrlForRegion(regionKey: string): string {
  const c = REGION_CENTERS[regionKey] ?? { lat: 36.8, lon: -121.9, label: "Central CA" };
  const params = new URLSearchParams({
    lat: String(c.lat),
    lon: String(c.lon),
    FcstType: "marine",
  });
  return `https://marine.weather.gov/MapClick.php?${params.toString()}`;
}

export function regionForecastLabel(regionKey: string): string {
  return REGION_CENTERS[regionKey]?.label ?? "California coast";
}

/** NWS point forecast (onshore) — useful for wind before leaving the dock */
export function pointForecastUrl(lat: number, lon: number): string {
  const params = new URLSearchParams({
    lat: String(lat),
    lon: String(lon),
    FcstType: "grid",
  });
  return `https://forecast.weather.gov/MapClick.php?${params.toString()}`;
}
