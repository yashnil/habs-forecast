import type { BiotoxinIndex, BiotoxinSet, BiotoxinSite } from "@/pilot/contracts";
import { cachedLoader, dayOrNull, isDay, isNum, isObj, isStr, numOrNull, oneOf, strList, strOrNull, type Obj } from "./validate";

/**
 * `${dataBase}/biotoxin/index.json`: CDPH domoic-acid test results parsed from CDPH's PDFs by the
 * pipeline (pipeline/coastwatch_pipeline/sources/biotoxin.py; schema schemas/v1/biotoxin.schema.json).
 *
 * Measured tests, published by CDPH: not a forecast, and they speak only for the species, site and
 * date tested. A null value is "ND", below the reporting limit, or unreadable, never zero. Published
 * outside the manifest. Missing or invalid file -> null; a malformed site or set is dropped, never
 * half-shown, and nothing else on the page changes.
 */

/** Optional extensions the pipeline adds (documented in pilot_models.py); the contract fields are untouched. */
export type BiotoxinSetExt = BiotoxinSet & { values_raw?: string[]; meat_values_ppm?: (number | null)[] | null; composite_of?: number | null; note?: string | null };
export type BiotoxinSiteExt = Omit<BiotoxinSite, "sets"> & { sets: BiotoxinSetExt[]; source_url?: string | null };
export type BiotoxinIndexExt = Omit<BiotoxinIndex, "sites"> & {
  sites: BiotoxinSiteExt[];
  additional_sources?: BiotoxinIndex["source"][];
  errors?: string[];
};

const KINDS = ["routine", "retest", "unknown"] as const;
const TISSUES = ["viscera", "meat", "whole", "unknown"] as const;
const valueList = (x: unknown): x is (number | null)[] => Array.isArray(x) && x.every((v) => v === null || isNum(v));

function source(x: unknown): BiotoxinIndex["source"] | null {
  if (!isObj(x) || !isStr(x.name) || !isStr(x.url) || !dayOrNull(x.as_of) || !isStr(x.fetched_at) || !isStr(x.sha256)) return null;
  return { name: x.name, url: x.url, as_of: x.as_of, fetched_at: x.fetched_at, sha256: x.sha256 };
}

function set(x: unknown): BiotoxinSetExt | null {
  if (!isObj(x) || !isDay(x.date) || !oneOf(x.kind, KINDS) || !oneOf(x.tissue, TISSUES) || !valueList(x.values_ppm) || x.values_ppm.length === 0) return null;
  if (!isNum(x.nd_count) || x.nd_count < 0 || !numOrNull(x.avg_ppm) || !numOrNull(x.max_ppm) || !numOrNull(x.pct_over_action) || !numOrNull(x.action_level_ppm)) return null;
  const out: BiotoxinSetExt = {
    date: x.date, kind: x.kind, tissue: x.tissue, values_ppm: x.values_ppm, nd_count: x.nd_count,
    avg_ppm: x.avg_ppm, max_ppm: x.max_ppm, pct_over_action: x.pct_over_action, action_level_ppm: x.action_level_ppm,
  };
  if (Array.isArray(x.values_raw) && x.values_raw.every((v) => typeof v === "string")) out.values_raw = x.values_raw as string[];
  if (valueList(x.meat_values_ppm) && x.meat_values_ppm.length === x.values_ppm.length) out.meat_values_ppm = x.meat_values_ppm;
  if (isNum(x.composite_of)) out.composite_of = x.composite_of;
  if (isStr(x.note)) out.note = x.note;
  return out;
}

function site(x: unknown): BiotoxinSiteExt | null {
  if (!isObj(x) || !isStr(x.id) || !strOrNull(x.bma) || !strOrNull(x.bma_label) || !strOrNull(x.port_label) || !isStr(x.site_label) || !isStr(x.species)) return null;
  if (!numOrNull(x.lat) || !numOrNull(x.lon) || !Array.isArray(x.informs_places) || !(x.informs_places.length === 0 || strList(x.informs_places)) || !Array.isArray(x.sets)) return null;
  const sets = x.sets.map(set).filter((s): s is BiotoxinSetExt => !!s).sort((a, b) => b.date.localeCompare(a.date));
  if (!sets.length) return null;
  return {
    id: x.id, bma: x.bma, bma_label: x.bma_label, port_label: x.port_label, site_label: x.site_label, species: x.species,
    lat: x.lat, lon: x.lon, informs_places: x.informs_places as string[], sets, source_url: isStr(x.source_url) ? x.source_url : null,
  };
}

/** The index as published, or null when it is not a version-1 BiotoxinIndex. */
export function parseBiotoxin(j: unknown): BiotoxinIndexExt | null {
  if (!isObj(j) || j.version !== 1 || !isStr(j.generated_at) || !Array.isArray(j.sites) || !Array.isArray(j.action_levels) || !Array.isArray(j.parse_warnings)) return null;
  const src = source(j.source);
  if (!src) return null;
  const levels = j.action_levels.filter((a): a is Obj => isObj(a) && isStr(a.species) && isStr(a.tissue) && isNum(a.ppm)).map((a) => ({ species: a.species as string, tissue: a.tissue as string, ppm: a.ppm as number }));
  return {
    version: 1,
    generated_at: j.generated_at,
    source: src,
    additional_sources: Array.isArray(j.additional_sources) ? j.additional_sources.map(source).filter((s): s is BiotoxinIndex["source"] => !!s) : [],
    action_levels: levels,
    sites: j.sites.map(site).filter((s): s is BiotoxinSiteExt => !!s),
    parse_warnings: j.parse_warnings.filter((w): w is string => typeof w === "string"),
    errors: Array.isArray(j.errors) ? j.errors.filter((e): e is string => typeof e === "string") : [],
  };
}

/** `${dataBase}/biotoxin/index.json`, fetched once per base; never throws (failure -> null). */
export const loadBiotoxin = cachedLoader("biotoxin/index.json", parseBiotoxin);

/** Sites whose results inform a pilot place (e.g. "port-592"), newest set first. */
export const sitesForPlace = (idx: BiotoxinIndexExt | null | undefined, placeId: string): BiotoxinSiteExt[] => (idx?.sites ?? []).filter((s) => s.informs_places.includes(placeId));

/** Samples in a set above its action level (as published); null when the set has no action level. */
export const overAction = (s: BiotoxinSet): number | null => (s.action_level_ppm == null ? null : s.values_ppm.filter((v) => v != null && v > s.action_level_ppm!).length);
