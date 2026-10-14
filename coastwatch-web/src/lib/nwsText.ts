import { distanceKm, type MarineAlert, type MarineForecast } from "@/lib/conditions";

/**
 * NWS coastal waters forecasts as NWS publishes them in plain text on tgftp.nws.noaa.gov (no
 * robots.txt; the host api.weather.gov, whose robots.txt disallows automated clients, is not used).
 *
 * The zone is never guessed from a point: each place CoastWatch knows (CDFW ports and CDPH harvest
 * sites) is assigned the inner coastal zone (out to 10 NM, or the named bay) that contains it by the
 * zone's own NWS description, checked on 2026-10-14 against each file's header:
 *   PZZ450 Pt. St. George to Cape Mendocino · PZZ455 Cape Mendocino to Pt. Arena ·
 *   PZZ540 Point Arena to Point Reyes · PZZ530 San Francisco Bay north of the Bay Bridge ·
 *   PZZ545 Point Reyes to Pigeon Point · PZZ535 Monterey Bay · PZZ645 Point Piedras Blancas to Point Sal ·
 *   PZZ650 East Santa Barbara Channel (Pt. Conception to Pt. Mugu) · PZZ655 Inner waters Point Mugu
 *   to San Mateo Pt. · PZZ740 San Mateo Point to the Mexican border, out to 10 NM.
 * A chosen point farther than ANCHOR_KM from every such place gets no zone (the card links to NWS).
 */
export const ANCHOR_KM = 20;
export const ZONE_ANCHORS: { name: string; lat: number; lon: number; zone: string }[] = [
  { name: "Crescent City", lat: 41.74901, lon: -124.20398, zone: "PZZ450" },
  { name: "Crescent Beach", lat: 41.71, lon: -124.143, zone: "PZZ450" },
  { name: "Trinidad", lat: 41.0593, lon: -124.14314, zone: "PZZ450" },
  { name: "Clam Beach", lat: 41.008, lon: -124.115, zone: "PZZ450" },
  { name: "Eureka", lat: 40.79834, lon: -124.16376, zone: "PZZ450" },
  { name: "Shelter Cove", lat: 40.0255, lon: -124.0664, zone: "PZZ455" },
  { name: "Fort Bragg", lat: 39.42518, lon: -123.80376, zone: "PZZ455" },
  { name: "Point Arena", lat: 38.9088, lon: -123.69308, zone: "PZZ540" },
  { name: "Bodega Bay", lat: 38.33325, lon: -123.04807, zone: "PZZ540" },
  { name: "Tomales Bay", lat: 38.205, lon: -122.931, zone: "PZZ540" },
  { name: "San Francisco", lat: 37.80865, lon: -122.41551, zone: "PZZ530" },
  { name: "Princeton / Half Moon Bay", lat: 37.5037, lon: -122.48185, zone: "PZZ545" },
  { name: "Santa Cruz", lat: 36.97412, lon: -122.03081, zone: "PZZ535" },
  { name: "Moss Landing", lat: 36.8044, lon: -121.78691, zone: "PZZ535" },
  { name: "Monterey", lat: 36.60024, lon: -121.89469, zone: "PZZ535" },
  { name: "Morro Bay", lat: 35.36581, lon: -120.84991, zone: "PZZ645" },
  { name: "Avila / Port San Luis", lat: 35.17498, lon: -120.75519, zone: "PZZ645" },
  { name: "Santa Barbara", lat: 34.40722, lon: -119.68876, zone: "PZZ650" },
  { name: "Goleta Pier", lat: 34.413, lon: -119.828, zone: "PZZ650" },
  { name: "Ventura", lat: 34.27834, lon: -119.29318, zone: "PZZ650" },
  { name: "Port Hueneme", lat: 34.14779, lon: -119.19512, zone: "PZZ650" },
  { name: "San Pedro", lat: 33.73586, lon: -118.2923, zone: "PZZ655" },
  { name: "Newport Beach", lat: 33.61546, lon: -117.93063, zone: "PZZ655" },
  { name: "Dana Point", lat: 33.46698, lon: -117.69812, zone: "PZZ655" },
  { name: "Oceanside", lat: 33.19536, lon: -117.38484, zone: "PZZ740" },
  { name: "San Diego", lat: 32.71512, lon: -117.13414, zone: "PZZ740" },
];

/** The coastal zone for waters within ANCHOR_KM of a known place, or null. */
export function zoneNear(lat: number, lon: number): { zone: string; place: string; km: number } | null {
  let best: { zone: string; place: string; km: number } | null = null;
  for (const a of ZONE_ANCHORS) {
    const km = distanceKm(lat, lon, a.lat, a.lon);
    if (km <= ANCHOR_KM && (!best || km < best.km)) best = { zone: a.zone, place: a.name, km };
  }
  return best;
}

export const zoneTextUrl = (zone: string) => `https://tgftp.nws.noaa.gov/data/forecasts/marine/coastal/pz/${zone.toLowerCase()}.txt`;
export const zonePageUrl = (zone: string) => `https://forecast.weather.gov/shmrn.php?mz=${zone.toLowerCase()}`;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "833 PM PDT Tue Oct 13 2026" → an ISO instant (Pacific daylight or standard time only). */
export function parseIssued(s: string): string | null {
  const m = s.match(/^(\d{1,2})(\d{2}) (AM|PM) (PDT|PST) \w{3} (\w{3}) (\d{1,2}) (\d{4})$/);
  if (!m) return null;
  let h = Number(m[1]) % 12;
  if (m[3] === "PM") h += 12;
  const mon = MONTHS.indexOf(m[5]);
  if (mon < 0) return null;
  const off = m[4] === "PDT" ? 7 : 8;
  return new Date(Date.UTC(Number(m[7]), mon, Number(m[6]), h + off, Number(m[2]))).toISOString();
}

/**
 * One zone's forecast from an NWS coastal waters text product: its name, issue time, the periods
 * (".TONIGHT...", ".WED...") and any headlines ("...SMALL CRAFT ADVISORY IN EFFECT..."), quoted as
 * published. Returns null when the product does not contain the zone.
 */
export function parseCoastalText(zone: string, text: string): { forecast: MarineForecast; headlines: string[] } | null {
  const lines = text.replace(/\r/g, "").split("\n");
  const start = lines.findIndex((l) => l.startsWith(`${zone.toUpperCase()}-`));
  if (start < 0) return null;
  const end = lines.findIndex((l, i) => i > start && l.trim() === "$$");
  const block = lines.slice(start + 1, end < 0 ? undefined : end);
  // the zone's name runs until the line ending with "-"; the issue time follows it
  let i = 0;
  const name: string[] = [];
  while (i < block.length && !name.join(" ").endsWith("-")) name.push(block[i++].trim());
  const issued = block[i++]?.trim() ?? "";
  const body = block.slice(i).join("\n");
  const headlines = [...body.matchAll(/^\.\.\.([^.][\s\S]*?)\.\.\.\s*$/gm)].map((m) => m[1].replace(/\s+/g, " ").trim()).filter((h) => !/^\w+$/.test(h));
  const periods = [...body.matchAll(/^\.([A-Z][A-Z ]+?)\.\.\.([\s\S]*?)(?=^\.[A-Z][A-Z ]+?\.\.\.|(?![\s\S]))/gm)].map((m) => ({
    name: m[1].trim().replace(/\b(\w)(\w*)/g, (_, a: string, b: string) => a + b.toLowerCase()),
    text: m[2].replace(/\s+/g, " ").trim(),
  }));
  if (!periods.length) return null;
  return { forecast: { zone: zone.toUpperCase(), zoneName: name.join(" ").replace(/-$/, "").trim(), updated: parseIssued(issued), periods: periods.slice(0, 4) }, headlines };
}

/** Headlines as alerts (the event is the headline's first clause, e.g. "SMALL CRAFT ADVISORY IN EFFECT ..."). */
export function headlineAlerts(headlines: string[], url: string): MarineAlert[] {
  return headlines.map((h) => {
    const event = h.match(/^(.*?)\s+(IN EFFECT|REMAINS IN EFFECT|NOW IN EFFECT|CANCELLED|EXPIRES?)\b/)?.[1] ?? h;
    return { event: event.charAt(0) + event.slice(1).toLowerCase(), headline: h.charAt(0) + h.slice(1).toLowerCase(), onset: null, ends: null, url };
  });
}
