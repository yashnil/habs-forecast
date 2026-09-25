/**
 * NASA GIBS satellite chlorophyll (open WMTS).
 * MODIS Aqua L3S CHL 8-day was removed from the "best" endpoint; we use VIIRS NOAA-20 daily Chl-a.
 */

export const GIBS_SATELLITE = {
  layerId: "VIIRS_NOAA20_Chlorophyll_a",
  /** Must match WMTS GetCapabilities TileMatrixSet for this layer */
  tileMatrixSet: "GoogleMapsCompatible_Level7",
  wmtsCapsUrl:
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/1.0.0/WMTSCapabilities.xml",
  /** Official color scale for the map layer */
  legendHorizontalSvg:
    "https://gibs.earthdata.nasa.gov/legends/VIIRS_NOAA20_Chlorophyll_a_H.svg",
} as const;

/** NASA PACE ocean color — often fills turbid bays / nearshore gaps where VIIRS L3 is masked. */
export const GIBS_PACE_CHL = {
  layerId: "OCI_PACE_Chlorophyll_a",
  tileMatrixSet: "GoogleMapsCompatible_Level7",
  legendHorizontalSvg:
    "https://gibs.earthdata.nasa.gov/legends/OCI_PACE_Chlorophyll_a_H.svg",
} as const;

/**
 * GIBS REST tiles use TileRow then TileCol; Mapbox substitutes `{z}/{y}/{x}` in that order.
 */
export function gibsChlorophyllTileUrlTemplate(
  layerId: string,
  dateYYYYMMDD: string,
  tileMatrixSet: string = GIBS_SATELLITE.tileMatrixSet,
): string {
  return `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${layerId}/default/${dateYYYYMMDD}/${tileMatrixSet}/{z}/{y}/{x}.png`;
}

/** Default Time (YYYY-MM-DD) for a layer block in GetCapabilities. */
export function parseGibsLayerDefaultDate(
  capsXml: string,
  layerId: string,
): string | null {
  const needle = `<ows:Identifier>${layerId}</ows:Identifier>`;
  const i = capsXml.indexOf(needle);
  if (i < 0) return null;
  const slice = capsXml.slice(i, i + 30_000);
  const m = slice.match(
    /<ows:Identifier>Time<\/ows:Identifier>[\s\S]*?<Default>([^<]+)<\/Default>/,
  );
  const raw = m?.[1]?.trim();
  if (!raw || !/^\d{4}-\d{2}-\d{2}$/.test(raw)) return null;
  return raw;
}

export function parseViirsNoaa20ChlDefaultDate(capsXml: string): string | null {
  return parseGibsLayerDefaultDate(capsXml, GIBS_SATELLITE.layerId);
}

export function parsePaceChlDefaultDate(capsXml: string): string | null {
  return parseGibsLayerDefaultDate(capsXml, GIBS_PACE_CHL.layerId);
}

/** If capabilities parsing fails: step back from UTC “today” (processing lag). */
export function fallbackSatelliteDate(daysBackFromUtc = 2): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - daysBackFromUtc);
  return d.toISOString().slice(0, 10);
}
