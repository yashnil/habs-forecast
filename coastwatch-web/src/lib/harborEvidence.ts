import type { Association } from "@/lib/applicability";
import type { Audience } from "@/lib/coast";
import { compass, stationName, type Conditions } from "@/lib/conditions";
import type { Freshness } from "@/lib/freshness";
import { ageWords, fmtMeasured, fmtPct, EVIDENCE_RADIUS_KM, NEIGHBOURHOOD_KM, RECENT_DAYS, type Neighbourhood, type StationEvidence } from "@/lib/hab";
import type { StationLite } from "@/lib/data";
import type { Verification } from "@/lib/official";
import { runAgeWords } from "@/lib/charmDays";
import { recordNamesSpecies, type Species } from "@/content/species";

/**
 * The harbor evidence graph: everything Neritic knows about one place, as typed nodes that each
 * carry how they are known (measured, retrieved from a satellite, published as a forecast,
 * reconstructed, or Neritic's own experiment), where and when they are true, how far they are
 * from the place and whether that distance lets them describe it, their freshness and their
 * caveats; plus the gaps, said as gaps; plus deterministic claims that cite the nodes they rest on.
 *
 * It is pure: the page and the API route feed it the same readings (shore stations, C-HARM cells,
 * the satellite pixel, live conditions, the notices) and get the same graph, so a sentence on the
 * page, an API answer and, later, an assistant all rest on the same evidence. Nothing here
 * interpolates, scores risk or speaks of safety.
 */

export type EvidenceClass = "measurement" | "satellite_retrieval" | "published_forecast" | "reconstruction" | "experimental";

/** How well a reading at `km` from the place can stand for the place, by the variable's own scale. */
export type Representative = "at" | "near" | "context" | "far";

export type EvidenceNode = {
  id: string;
  class: EvidenceClass;
  variable: string;
  units: string;
  value: number | null;
  /** the value as the briefing says it ("4,900 cells/L") */
  text: string;
  source: { id: string; name: string; url?: string | null; product?: string; version?: string };
  time: { observed?: string | null; valid?: string | null; issued?: string | null; ageDays: number | null };
  support: { kind: "point" | "cell" | "cells" | "pixel" | "pixels" | "zone"; nativeM?: number | null; radiusKm?: number; n?: number };
  where: { lat: number | null; lon: number | null; name: string | null; km: number; bearing: string | null; representative: Representative; setting?: Setting | null };
  freshness: { state: Freshness["state"] | "unknown"; note?: string };
  quality: { flags: string[] };
  lineage: string[];
  caveats: string[];
};

export type Gap = { id: string; variable: string; reason: string; nearest?: { name: string; km: number; date?: string | null } | null };

export type Claim = { id: string; text: string; supports: string[]; kind: "measured" | "model" | "satellite" | "conditions" | "official" | "gap" };

export type EvidenceGraph = {
  place: { name: string; lat: number; lon: number };
  generatedAt: string;
  window: { from: string; to: string };
  nodes: EvidenceNode[];
  gaps: Gap[];
  claims: Claim[];
  /** what the graph does not try to say */
  limits: string[];
};

// ---------------------------------------------------------------- representativeness, by variable

/**
 * Distances within which a reading can stand for a place, by variable. They are not one threshold:
 * - Pseudo-nitzschia counts and particulate domoic acid are sampled at piers weekly; bloom patches
 *   along this coast are of order 10 km and the nowcast/forecast skill study (Anderson et al. 2016)
 *   matched model cells to piers within about 10–15 km. A pier farther than EVIDENCE_RADIUS_KM
 *   (60 km) is regional context at best.
 * - Satellite chlorophyll pixels (300–750 m) describe the pixel; a median within 5 km is "near"
 *   (lib/chlNear's place rule), with shore-adjacent pixels set aside.
 * - C-HARM cells are 3 km; the waters' own cell is "at", offshore cells within 15 km are context
 *   only (the published port rule), and a nearest cell beyond that is named with its distance.
 * - Wind and seas are synoptic (a buoy within 30 km describes the same weather, though not the
 *   same exposure); water temperature from a harbor tide station is local (2 km), from a buoy it
 *   is "near" to 20 km.
 */
export const REPRESENTATIVE_KM: Record<string, { at: number; near: number; context: number }> = {
  pn_cells: { at: 2, near: 15, context: EVIDENCE_RADIUS_KM },
  pda: { at: 2, near: 15, context: EVIDENCE_RADIUS_KM },
  chl_satellite: { at: 0.5, near: 5, context: 15 },
  charm_cell: { at: 1.6, near: 1.6, context: NEIGHBOURHOOD_KM },
  wind: { at: 5, near: 30, context: 60 },
  waves: { at: 5, near: 30, context: 60 },
  water_temp: { at: 2, near: 20, context: 60 },
  tide: { at: 5, near: 40, context: 80 },
};

export function representative(variable: keyof typeof REPRESENTATIVE_KM, km: number): Representative {
  const r = REPRESENTATIVE_KM[variable];
  return km <= r.at ? "at" : km <= r.near ? "near" : km <= r.context ? "context" : "far";
}

/**
 * The setting of a shore station, from its name: an open pier or wharf samples exposed coastal
 * water; a station inside a harbor or lagoon samples sheltered water that can differ from the
 * coast outside. A heuristic, said as one; it never changes a value.
 */
export type Setting = "exposed-pier" | "sheltered-harbor" | "offshore" | "unclassified";
export function stationSetting(name: string): Setting {
  const n = name.toLowerCase();
  if (/\b(harbor|harbour|lagoon|slough|marina|back bay|inner)\b/.test(n)) return "sheltered-harbor";
  if (/\b(pier|wharf|jetty|point|beach|cove)\b/.test(n)) return "exposed-pier";
  if (/\b(buoy|nm|offshore|canyon)\b/.test(n)) return "offshore";
  return "unclassified";
}

// ---------------------------------------------------------------- inputs

export type OutlookDayLite = { lead: number; valid: string | null; pda: Neighbourhood | null; pn: Neighbourhood | null };

export type SatLite =
  | { value: number; date: string | null; km: number; sensor: string; layerId: string; basis: { n: number; min: number; max: number; radiusKm: number } | null; aside: { value: number; reason: "shore" | "spike" } | null }
  | { value: null; sensor: string; windowDays: number; flagged: boolean; radiusKm: number }
  | null;

export type EvidenceInputs = {
  place: { name: string; lat: number; lon: number };
  now: Date;
  /** the trip day the model is read for */
  tripDate: string | null;
  measured: { cells: StationEvidence | null; toxin: StationEvidence | null; nearest: { station: StationLite; km: number; date: string; ageDays: number } | null } | null;
  measuredError?: string | null;
  model: { days: OutlookDayLite[]; issued: string | null; fresh: Freshness | null; version?: string } | null | "error";
  sat: SatLite | "error";
  conditions: Conditions | null;
  official: { assocs: Association[]; verification: Verification | null; audience: Audience; species: Species } | null;
};

const km = (k: number) => (k < 10 ? k.toFixed(1) : String(Math.round(k)));
const DAY = 86400000;

// ---------------------------------------------------------------- the graph

export function harborEvidence(x: EvidenceInputs): EvidenceGraph {
  const nodes: EvidenceNode[] = [];
  const gaps: Gap[] = [];
  const claims: Claim[] = [];
  const P = x.place;
  const today = x.now.toISOString().slice(0, 10);

  // ---- shore measurements (CalHABMAP): the only toxin measurements that exist near shore
  const station = (e: StationEvidence, variable: "pn_cells" | "pda"): EvidenceNode => {
    const total = e.kind === "cells" ? e.values.reduce((a, v) => a + v.value, 0) : e.values[0].value;
    const rep = representative(variable, e.km);
    const setting = stationSetting(e.station.name);
    return {
      id: `${variable}:${e.station.station_id}`,
      class: "measurement",
      variable: variable === "pda" ? "particulate domoic acid in seawater" : "Pseudo-nitzschia cells",
      units: variable === "pda" ? "ng/mL" : "cells/L",
      value: total,
      text: fmtMeasured(variable === "pda" ? "pDA" : "pn_seriata", total),
      source: { id: "calhabmap", name: "CalHABMAP via SCCOOS ERDDAP", url: e.station.source_url, product: "HABs shore station" },
      time: { observed: e.date, ageDays: e.ageDays },
      support: { kind: "point" },
      where: { lat: e.station.lat, lon: e.station.lon, name: e.station.name, km: e.km, bearing: e.bearing, representative: rep, setting },
      freshness: { state: e.ageDays <= RECENT_DAYS ? "current" : e.ageDays <= 90 ? "stale" : "historical" },
      quality: { flags: [] },
      lineage: ["sampled at the pier", "counted or assayed in a laboratory", "published on SCCOOS ERDDAP", "read by Neritic's pipeline"],
      caveats: ["Seawater, not seafood: only CDPH and CDFW test shellfish and fish.", ...(setting === "sheltered-harbor" ? ["Sampled inside a harbor: sheltered water can differ from the coast outside."] : setting === "exposed-pier" ? ["Sampled at an open pier: exposed coastal water, which can differ from sheltered harbor water."] : [])],
    };
  };
  if (x.measuredError) gaps.push({ id: "measured", variable: "shore measurements", reason: `Shore measurements could not be loaded (${x.measuredError}).` });
  else if (x.measured) {
    const used: EvidenceNode[] = [];
    for (const [e, v] of [
      [x.measured.cells, "pn_cells"],
      [x.measured.toxin, "pda"],
    ] as const) {
      if (!e) continue;
      const n = station(e, v);
      nodes.push(n);
      if (e.near && e.ageDays <= RECENT_DAYS) used.push(n);
      else gaps.push({ id: `${v}-recent`, variable: n.variable, reason: e.near ? `No ${n.variable} sample within ${EVIDENCE_RADIUS_KM} km in the last ${RECENT_DAYS} days; the latest nearby is ${ageWords(e.ageDays)} at ${e.station.name} (${km(e.km)} km).` : `No shore station within ${EVIDENCE_RADIUS_KM} km measures ${n.variable}; the nearest that does, ${e.station.name}, is ${km(e.km)} km away, too far to describe these waters.`, nearest: { name: e.station.name, km: e.km, date: e.date } });
    }
    const best = [...used].sort((a, b) => (b.time.observed ?? "").localeCompare(a.time.observed ?? "") || a.where.km - b.where.km)[0];
    if (best) {
      const c = x.measured.nearest;
      const closer = c && c.station.station_id !== (best.id.split(":")[1] ?? "") && c.km < best.where.km;
      claims.push({
        id: "measured",
        kind: "measured",
        text: `${best.where.name} reported ${best.text.replace(/^(\d)/, "$1")} ${best.variable === "Pseudo-nitzschia cells" ? "Pseudo-nitzschia" : "particulate domoic acid"} on ${longDate(best.time.observed!)}, ${km(best.where.km)} km ${best.where.bearing} of ${P.name}.${closer ? ` ${c.station.name} is closer (${km(c.km)} km) but has no comparable sample since ${longDate(c.date, x.now)}.` : ""}`,
        supports: [best.id],
      });
    } else if (!x.measured.cells && !x.measured.toxin) gaps.push({ id: "measured-none", variable: "shore measurements", reason: "No shore station measures bloom algae or toxin near these waters." });
  }

  // ---- C-HARM (published forecast): the waters' own cell, or the offshore cells as context
  if (x.model === "error") gaps.push({ id: "model", variable: "C-HARM", reason: "No C-HARM run could be read." });
  else if (x.model) {
    // the trip day only: a day the run does not reach is a gap, never the nearest other day
    const day = x.tripDate ? (x.model.days.find((d) => d.valid === x.tripDate) ?? null) : (x.model.days[0] ?? null);
    const nb = day?.pda ?? null;
    const freshNote = x.model.fresh && x.model.fresh.state !== "current" ? runAgeWords(x.model.fresh) : null;
    const base = {
      class: "published_forecast" as const,
      variable: "probability that particulate domoic acid exceeds 500 ng/L",
      units: "probability",
      source: { id: "charm", name: "NOAA C-HARM", url: "https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.html", product: "C-HARM", version: x.model.version ?? "3.1" },
      time: { valid: day?.valid ?? null, issued: x.model.issued, ageDays: x.model.fresh?.ageDays ?? null },
      freshness: { state: (x.model.fresh?.state ?? "unknown") as EvidenceNode["freshness"]["state"], ...(freshNote ? { note: freshNote } : {}) },
      quality: { flags: [] as string[] },
      lineage: ["NOAA's statistical model on satellite and ocean-model inputs", "published on NOAA CoastWatch ERDDAP", "read by Neritic's pipeline, values unchanged"],
      caveats: ["A probability for a 3 km cell, not a toxin measurement or a closure decision.", "A low probability does not mean an area is safe.", ...(freshNote ? [`NOAA's runs are often missing a day or more: ${freshNote}.`] : [])],
    };
    if (!day) gaps.push({ id: "model-day", variable: "C-HARM", reason: `NOAA's model does not reach ${x.tripDate ?? "that day"}.` });
    else if (nb?.here != null) {
      const n: EvidenceNode = { ...base, id: `charm:own:${day.valid}`, value: nb.here, text: fmtPct(nb.here), support: { kind: "cell", nativeM: 3000 }, where: { lat: P.lat, lon: P.lon, name: "these waters' own 3 km cell", km: 0, bearing: null, representative: "at" } };
      nodes.push(n);
      claims.push({ id: "model", kind: "model", text: `NOAA's C-HARM gives the 3 km cell containing ${P.name} a ${fmtPct(nb.here)} chance that particulate domoic acid exceeds its threshold on ${longDate(day.valid!)}${freshNote ? ` (${freshNote})` : ""}. A probability, not a measurement.`, supports: [n.id] });
    } else if (nb && nb.n > 0) {
      const n: EvidenceNode = { ...base, id: `charm:offshore:${day.valid}`, value: nb.median, text: `${fmtPct(nb.min)}–${fmtPct(nb.max)}`, support: { kind: "cells", nativeM: 3000, radiusKm: NEIGHBOURHOOD_KM, n: nb.n }, where: { lat: nb.nearest?.cell.lat ?? null, lon: nb.nearest?.cell.lon ?? null, name: `${nb.n} offshore cells within ${NEIGHBOURHOOD_KM} km`, km: nb.nearest?.km ?? NEIGHBOURHOOD_KM, bearing: null, representative: "context" }, caveats: [...base.caveats, "Offshore context only: none of these cells contains the waters."] };
      nodes.push(n);
      gaps.push({ id: "model-here", variable: "C-HARM", reason: `NOAA's C-HARM model has no valid estimate in these nearshore waters; its nearest cell with a value is ${km(nb.nearest!.km)} km out.`, nearest: { name: "nearest C-HARM cell", km: nb.nearest!.km } });
      claims.push({ id: "model", kind: "gap", text: `NOAA's C-HARM model has no valid estimate in these nearshore waters. Its nearest cell is ${km(nb.nearest!.km)} km out; the ${nb.n} offshore cells within ${NEIGHBOURHOOD_KM} km read ${fmtPct(nb.min)}–${fmtPct(nb.max)} for ${longDate(day.valid!)}, context for the open water, not the harbor.`, supports: [n.id] });
    } else if (nb?.nearest) {
      gaps.push({ id: "model-here", variable: "C-HARM", reason: `These waters are outside NOAA's model: no cell within ${NEIGHBOURHOOD_KM} km. Its nearest cell, ${km(nb.nearest.km)} km away, reads ${fmtPct(nb.nearest.value)}; that describes that cell, not these waters.`, nearest: { name: "nearest C-HARM cell", km: nb.nearest.km } });
      claims.push({ id: "model", kind: "gap", text: `NOAA's C-HARM model has no estimate within ${NEIGHBOURHOOD_KM} km of ${P.name}; its nearest cell is ${km(nb.nearest.km)} km away.`, supports: [] });
    } else gaps.push({ id: "model-here", variable: "C-HARM", reason: "These waters are outside NOAA's model: no cell within 80 km." });
  }

  // ---- satellite chlorophyll (retrieval): algae of every kind, not toxin
  if (x.sat === "error") gaps.push({ id: "satellite", variable: "satellite chlorophyll", reason: "The satellite grid could not be read." });
  else if (x.sat && x.sat.value != null) {
    const s = x.sat;
    const age = s.date ? Math.floor((x.now.getTime() - Date.parse(`${s.date}T12:00:00Z`)) / DAY) : null;
    const n: EvidenceNode = {
      id: `chl:${s.layerId}`,
      class: "satellite_retrieval",
      variable: "chlorophyll-a",
      units: "mg/m³",
      value: s.value,
      text: `${Number(s.value.toPrecision(2))} mg/m³`,
      source: { id: "satellite_chl", name: `NOAA CoastWatch, ${s.sensor}`, product: s.layerId },
      time: { observed: s.date, ageDays: age },
      support: s.basis ? { kind: "pixels", radiusKm: s.basis.radiusKm, n: s.basis.n, nativeM: /300/.test(s.sensor) ? 300 : 750 } : { kind: "pixel", nativeM: /300/.test(s.sensor) ? 300 : 750 },
      where: { lat: null, lon: null, name: s.basis ? `median of ${s.basis.n} clear pixels within ${s.basis.radiusKm} km` : "the pixel at these waters", km: s.km, bearing: null, representative: representative("chl_satellite", s.km) },
      freshness: { state: age == null ? "unknown" : age <= 3 ? "current" : age <= 7 ? "stale" : "historical" },
      quality: { flags: s.aside ? [`a ${s.aside.reason === "shore" ? "shore-adjacent" : "spike"} pixel (${Number(s.aside.value.toPrecision(2))}) was set aside`] : [] },
      lineage: ["ocean-colour retrieval by NOAA CoastWatch", "latest clear pixel within 7 days", "median within 5 km when the pixel is unclear"],
      caveats: ["Algae of every kind, not toxin: high chlorophyll does not confirm a harmful bloom, and low does not rule one out."],
    };
    nodes.push(n);
    claims.push({ id: "satellite", kind: "satellite", text: `Satellite chlorophyll ${s.basis ? `near ${P.name} (median of ${s.basis.n} clear pixels within ${s.basis.radiusKm} km)` : `at ${P.name}`} was ${n.text}${s.date ? ` on ${longDate(s.date)}` : ""} (${s.sensor}). Algae, not toxin.`, supports: [n.id] });
  } else if (x.sat) gaps.push({ id: "satellite", variable: "satellite chlorophyll", reason: x.sat.flagged ? `No open-water chlorophyll reading within ${x.sat.radiusKm} km: the pixels there are shore-adjacent or flagged.` : `No clear satellite view within ${x.sat.radiusKm} km in the last ${x.sat.windowDays} days: cloud, fog or no overpass, not low chlorophyll.` });

  // ---- live conditions (measurements and an official forecast), context never a HAB forecast
  const c = x.conditions;
  if (c) {
    if (c.buoy.status === "ok" && c.buoy.station) {
      const b = c.buoy.data;
      const st = c.buoy.station;
      const setting = stationSetting(st.name);
      const add = (key: keyof typeof b, variable: string, units: string, rep: keyof typeof REPRESENTATIVE_KM, fmt: (v: number) => string) => {
        const o = b[key];
        if (!o) return null;
        const n: EvidenceNode = {
          id: `buoy:${st.id}:${key}`,
          class: "measurement",
          variable,
          units,
          value: o.value,
          text: fmt(o.value),
          source: { id: "ndbc", name: "NOAA National Data Buoy Center", url: c.buoy.status === "ok" ? c.buoy.url : null, product: st.id },
          time: { observed: o.at, ageDays: 0 },
          support: { kind: "point" },
          where: { lat: st.lat ?? null, lon: st.lon ?? null, name: stationName(st.name), km: st.km, bearing: null, representative: representative(rep, st.km), setting },
          freshness: { state: "current" },
          quality: { flags: [] },
          lineage: ["measured at the station", "NDBC realtime2 text", "read live by Neritic's server"],
          caveats: [setting === "offshore" ? "An offshore buoy: seas at the harbor entrance can differ." : setting === "sheltered-harbor" ? "A station inside the harbor: conditions outside can differ." : "Wind and seas, not a HAB or fishing forecast."],
        };
        nodes.push(n);
        return n;
      };
      const wind = add("windKt", "wind", "kt", "wind", (v) => `${Math.round(v)} kt${b.windDirDeg ? ` ${compass(b.windDirDeg.value)}` : ""}`);
      const wave = add("waveFt", "significant wave height", "ft", "waves", (v) => `${v.toFixed(1)} ft${b.periodS ? ` at ${Math.round(b.periodS.value)} s` : ""}`);
      const temp = add("waterC", "water temperature", "°C", "water_temp", (v) => `${v.toFixed(1)} °C`);
      const parts = [wind && `wind ${wind.text}`, wave && `seas ${wave.text}`, temp && `water ${temp.text}`].filter(Boolean);
      if (parts.length) claims.push({ id: "buoy", kind: "conditions", text: `${stationName(st.name)} (${km(st.km)} km) now: ${parts.join(", ")}. Measured there, not at the dock.`, supports: [wind, wave, temp].filter((n): n is EvidenceNode => !!n).map((n) => n.id) });
    } else if (c.buoy.status !== "ok") gaps.push({ id: "buoy", variable: "wind and seas", reason: c.buoy.reason });
    if (c.marine.status === "ok" && c.marine.data.periods[0]) {
      const p = c.marine.data.periods[0];
      const n: EvidenceNode = {
        id: `nws:${c.marine.data.zone}`,
        class: "published_forecast",
        variable: "marine forecast",
        units: "",
        value: null,
        text: p.text,
        source: { id: "nws", name: "NOAA National Weather Service", url: c.marine.url, product: c.marine.data.zone },
        time: { issued: c.marine.data.updated, ageDays: null },
        support: { kind: "zone" },
        where: { lat: null, lon: null, name: c.marine.data.zoneName, km: 0, bearing: null, representative: "near" },
        freshness: { state: "current" },
        quality: { flags: [] },
        lineage: ["written by NWS forecasters for the whole zone", "read from NWS's text product"],
        caveats: ["A forecast for the whole marine zone, in NWS's words."],
      };
      nodes.push(n);
      claims.push({ id: "nws", kind: "conditions", text: `NWS ${p.name.toLowerCase()} for ${c.marine.data.zoneName}: “${p.text}”`, supports: [n.id] });
    }
  }

  // ---- official notices: carried with their verification state, never adjudicated here
  if (x.official) {
    const o = x.official;
    const applying = o.assocs.filter((a) => !a.dispute && a.tier === "geographic" && recordNamesSpecies(a.record.species, o.species));
    const disputed = o.assocs.filter((a) => a.dispute && recordNamesSpecies(a.record.species, o.species));
    const unverified = !o.verification || (o.verification.state !== "verified" && o.verification.state !== "unavailable");
    for (const a of [...disputed, ...applying]) {
      const url = a.record.sources.find((s) => s.url.startsWith("https://"))?.url ?? null;
      nodes.push({
        id: `official:${a.record.id}`,
        class: "measurement",
        variable: "official notice",
        units: "",
        value: null,
        text: a.record.title,
        source: { id: "official", name: a.record.agency, url, product: a.record.id },
        time: { issued: a.record.effective_date ?? null, ageDays: null },
        support: { kind: "zone" },
        where: { lat: null, lon: null, name: a.record.area.description ?? null, km: 0, bearing: null, representative: "at" },
        freshness: { state: a.dispute ? "stale" : "current", ...(a.dispute ? { note: `${a.dispute.agency} release ${a.dispute.release} appears to lift it; awaiting review` } : {}) },
        quality: { flags: unverified ? ["Neritic's transcription, not verified by a person"] : [] },
        lineage: ["agency page", "transcribed by Neritic's pipeline", unverified ? "awaiting human review" : "reviewed"],
        caveats: ["The agency's wording controls. A notice missing here does not mean open or safe."],
      });
    }
    const n = applying.length + disputed.length;
    claims.push({
      id: "official",
      kind: "official",
      text: n ? `${n} agency notice${n === 1 ? "" : "s"} ${n === 1 ? "names" : "name"} these waters${disputed.length ? ` (${disputed.length} appear${disputed.length === 1 ? "s" : ""} to have been lifted, awaiting review)` : ""}: ${[...disputed, ...applying].slice(0, 3).map((a) => `${a.record.agency}: ${a.record.title}`).join("; ")}. ${unverified ? "Neritic's copy, not verified by a person; the agency pages are the authority." : ""}`.trim() : "No notice in Neritic's copy names these waters. Not the same as open or safe: the copy can lag the agencies.",
      supports: [...disputed, ...applying].map((a) => `official:${a.record.id}`),
    });
  }

  return {
    place: P,
    generatedAt: x.now.toISOString(),
    window: { from: new Date(x.now.getTime() - 365 * DAY).toISOString().slice(0, 10), to: x.tripDate ?? today },
    nodes,
    gaps,
    claims,
    limits: [
      "No node is a seafood-safety statement, a closure decision or a catch prediction.",
      "No value is interpolated between cells or stations; distances are stated and representativeness is per variable.",
      "Neritic's own experimental estimates are not included until independently validated.",
    ],
  };
}

/** "September 30", or "August 27, 2025" when the date is not from the last 300 days of `now`. */
function longDate(d: string, now?: Date): string {
  const t = new Date(`${d.slice(0, 10)}T12:00:00Z`);
  const old = now ? now.getTime() - t.getTime() > 300 * DAY : false;
  return t.toLocaleDateString("en-US", { month: "long", day: "numeric", ...(old ? { year: "numeric" } : {}), timeZone: "UTC" });
}
