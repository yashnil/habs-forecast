/**
 * The catalog's example data: real values only, each traceable to a committed file (see each JSON's
 * `_source`). Nothing here is invented; the worksheet's entries are labelled as example entries.
 */
import type { PortEntry, Provenance, SeriesPoint } from "@/nx/ds";
import dated from "./data/dated.json";
import platesJson from "./data/plates.json";
import portsJson from "./data/ports.json";
import recordJson from "./data/record.json";
import snapshot from "./data/snapshot.json";

export const D = dated;
export const NOW = dated.now;
export const S = snapshot;

export interface PlateData {
  width: number;
  height: number;
  water: string;
  /** The shore only: the water rings without their tile seams. */
  coast: string;
  isobaths: Array<{ depth_m: number; pts: Array<[number, number]> }>;
}
export const plates = platesJson as unknown as {
  bay: PlateData & { places: Record<string, [number, number]> };
  chl: PlateData & { rowEdges: number[]; colEdges: number[]; grid: { asOf: string; sensor: string; units: string; source: string; rows: Array<Array<number | null>>; ages: Array<Array<number | null>> } };
};

export const ports = portsJson.ports as PortEntry[];
export const regions = portsJson.regions.map((r) => ({ id: r.id, label: r.label }));

type RawSeries = { label: string; units: string; points: Array<[string, number | null, string | null]> };
const rec = recordJson as unknown as { station: { name: string; source_url: string; retrieved_at: string; code: string }; series: Record<string, RawSeries> };
export const station = rec.station;

function toPoints(r: RawSeries): SeriesPoint[] {
  return r.points.map(([t, v, q]) => (q === "reported_zero" ? { t, v, flag: "reported0" as const, raw: "0" } : { t, v }));
}
export const record = {
  pDA: { ...rec.series.pDA, points: toPoints(rec.series.pDA) },
  pn: { ...rec.series.pn_seriata, points: toPoints(rec.series.pn_seriata) },
  chl: { ...rec.series.chl_extracted, points: toPoints(rec.series.chl_extracted) },
  temp: { ...rec.series.temp, points: toPoints(rec.series.temp) },
};

/** C-HARM 30-day nowcast medians at Moss Landing, as a model series (fraction to percent). */
export const charmSeries: SeriesPoint[] = snapshot.charm.last_30d_nowcast_medians.particulate_domoic.map(([t, v]) => ({ t: t as string, v: Math.round((v as number) * 100) }));

/** The crab test as CDPH printed it: censored values kept verbatim (T7). */
export const rockCrab = snapshot.cdph_crab_da.sites_for_port_592[1];
export const dungeness = snapshot.cdph_crab_da.sites_for_port_592[0];

const buoy = snapshot.buoy.waves;
export const P: Record<string, Provenance> = {
  waves: {
    kind: "measured",
    title: "Wave height, buoy 46236",
    value: `${buoy.wvht_ft} ft at ${buoy.period_dominant_s} s`,
    source: "NDBC buoy 46236",
    sourceLong: buoy.station,
    sourceUrl: `https://${buoy.source}`,
    date: buoy.asOf,
    retrieved: snapshot.buoy.asOf,
    place: "15.3 km from Moss Landing, at the outer Monterey Canyon",
    method: "Significant wave height and dominant period from the buoy's hourly record.",
    caveats: ["A wave buoy without wind; wind comes from NDBC 46092, 22 km away.", "One point at sea, not the harbour entrance."],
  },
  crab: {
    kind: "measured",
    title: "Domoic acid in Dungeness crab, Monterey Bay",
    value: `${dungeness.sets[0].max_ppm} ppm, highest of ${dungeness.sets[0].values_raw.length}`,
    source: "CDPH crab test",
    sourceLong: "CDPH CrabDAResults.pdf, Monterey Bay test area (BMA G)",
    sourceUrl: "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx",
    date: dungeness.sets[0].date,
    dateRole: "sampled",
    retrieved: snapshot.cdph_crab_da.fetched_at,
    place: "Monterey Bay test area, about 10 km west of Moss Landing",
    method: `Viscera of ${dungeness.sets[0].values_raw.length} crab; average ${dungeness.sets[0].avg_ppm} ppm, highest ${dungeness.sets[0].max_ppm} ppm. Action level 30 ppm.`,
    caveats: ["A test is a sample from a named site and day, not a closure decision. CDPH and CDFW decide.", "43 days old on Nov 10; CDPH tests on no fixed schedule."],
    stale: true,
  },
  season: {
    kind: "official",
    title: "Commercial Dungeness season, Zones 1 to 5",
    value: "Delayed to at least Jan 1, 2027",
    source: "CDFW",
    // design-lint-allow forbidden-word: CDFW's programme name, quoted as the agency writes it
    sourceLong: "California Department of Fish and Wildlife, Whale Safe Fisheries (RAMP)",
    sourceUrl: snapshot.fishery_status.records[0].source_url,
    date: snapshot.fishery_status.records[0].checked_at,
    dateRole: "checked",
    place: "California/Oregon border to Point Conception (Fishing Zones 1 to 5)",
    caveats: ["A whale-entanglement decision under RAMP, not a domoic acid closure.", "Next assessment Dec 15–18, then Jan 13–15, subject to change."],
  },
  charm: {
    kind: "model",
    title: "C-HARM, probability of particulate domoic acid over 500 ng/L",
    value: `${Math.round(snapshot.charm.leads[0].particulate_domoic.median * 100)} %`,
    source: "NOAA C-HARM v3.1, past run",
    sourceLong: "NOAA CoastWatch C-HARM v3.1 (wvcharmV3), median of 28 cells within 15 km",
    sourceUrl: "https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.html",
    date: snapshot.charm.asOf,
    dateRole: "issued",
    place: "Median of model cells within 15 km of Moss Landing; the harbour is inside the model's coastal gap",
    caveats: [snapshot.charm.must_say, "Every lead of this run is in the past (valid Nov 2–5)."],
    stale: true,
  },
  chl: {
    kind: "measured",
    title: "Chlorophyll near Moss Landing",
    value: `${snapshot.chlorophyll.near_port.median_mg_m3_within_15km} mg/m³`,
    source: "Sentinel-3 OLCI",
    sourceLong: snapshot.chlorophyll.near_port.sensor,
    date: snapshot.chlorophyll.near_port.asOf,
    place: "Median of 4,626 pixels within 15 km",
    caveats: [snapshot.chlorophyll.must_say],
  },
  tide: {
    kind: "prediction",
    title: "High tide, General Fish Company Pier",
    value: `${snapshot.tide.next_events_after_asOf[0].ft.toFixed(1)} ft`,
    source: "NOAA CO-OPS 9413617",
    sourceLong: snapshot.tide.station,
    sourceUrl: "https://tidesandcurrents.noaa.gov/stationhome.html?id=9413617",
    date: `${snapshot.tide.next_events_after_asOf[0].t.replace(" ", "T")}:00-08:00`,
    dateRole: "predicted",
    caveats: [snapshot.tide.note],
  },
  forecast: {
    kind: "prediction",
    title: "NWS coastal waters forecast, Monterey Bay",
    value: "Seas 4 to 6 ft",
    source: "NWS forecast PZZ535",
    sourceLong: snapshot.marine_forecast.source,
    sourceUrl: "https://tgftp.nws.noaa.gov/data/forecasts/marine/coastal/pz/pzz535.txt",
    date: snapshot.marine_forecast.asOf,
    dateRole: "issued",
  },
};
