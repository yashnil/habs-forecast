import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Manifest } from "@/generated/schema";
import { artifactUrl, charmLayer, satelliteLatest, tileIndexUrl, CHARM_LEADS, type SatProduct } from "@/lib/layers";
import { chooseHour } from "@/lib/currents";
import { supportsGridTiles } from "@/lib/gridTiles";
import { CHL_PALETTE, SST_PALETTE, derivedOutlook, derivedSst, parseDerived, sstIsAnalysis, type DerivedIndex } from "@/lib/derived";
import { dataBaseUrl, loadData, type DataLoad } from "@/lib/data";
import { displayPalette, gradientCss } from "@/lib/palette";
import { labelledGeometry } from "@/lib/officialGeometry";
import { pacificDate } from "@/lib/time";
import { dateSpan, isPastRun, shortDate } from "./caption";
import type { StageData } from "./types";

const CHL_TICKS = ["0.05", "0.3", "1.6", "9", "50"];

async function derivedIndex(base: string): Promise<DerivedIndex | null> {
  try {
    if (/^https?:\/\//.test(base)) {
      const r = await fetch(`${base}/derived/index.json`, { next: { revalidate: 300 } });
      return r.ok ? parseDerived(await r.json()) : null;
    }
    const dir = process.env.CW_DATA_DIR || path.join(process.cwd(), "public", base);
    return parseDerived(JSON.parse(await readFile(path.join(dir, "derived/index.json"), "utf8")));
  } catch {
    return null;
  }
}

function chlorophyll(m: Manifest, baseUrl: string): StageData["chlorophyll"] {
  for (const product of ["multi", "olci300", "viirs750"] as SatProduct[]) {
    const l = satelliteLatest(m, product);
    const t = l?.tiles;
    if (!l || !t) continue;
    const palette = l.palette ?? CHL_PALETTE;
    return {
      raster: { id: l.layer_id, template: t.relative ? artifactUrl(baseUrl, t.url_template) : t.url_template, minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat, indexUrl: tileIndexUrl(baseUrl, t) },
      caption: { what: "Chlorophyll", kind: "observed", date: dateSpan(l.composite?.oldest_observed_date, l.composite?.newest_observed_date ?? l.time.observed_date), ramp: { css: gradientCss(displayPalette(palette)), labels: CHL_TICKS, unit: "mg/m³" } },
    };
  }
  return null;
}

function charm(m: Manifest, baseUrl: string, today: string): StageData["charm"] {
  for (const lead of CHARM_LEADS) {
    const l = charmLayer(m, "particulate_domoic", lead) ?? charmLayer(m, "pseudo_nitzschia", lead);
    if (!l?.image) continue;
    const last = CHARM_LEADS.map((k) => charmLayer(m, l.variable as string, k)?.time.valid_date).filter(Boolean).sort().pop();
    return {
      raster: { id: l.layer_id, url: artifactUrl(baseUrl, l.image.url), corners: l.image.corners_lnglat, grid: supportsGridTiles(l) ? { layer: l, url: artifactUrl(baseUrl, l.grid!.url) } : null },
      caption: { what: "C-HARM probability", kind: "model", date: shortDate(l.time.valid_date ?? ""), past: isPastRun(last, today), ramp: l.palette ? { css: gradientCss(l.palette), labels: ["0%", "25%", "50%", "75%", "100%"] } : null },
    };
  }
  return null;
}

/** Only the geometry and names the stage draws, not the whole registry. */
function slimOfficial(fc: GeoJSON.FeatureCollection | null): GeoJSON.FeatureCollection | null {
  if (!fc) return null;
  return { type: "FeatureCollection", features: fc.features.map((f) => ({ ...f, properties: { kind: f.properties?.kind, label: f.properties?.label, record_ids: f.properties?.record_ids } })) };
}

/**
 * What OceanStage needs, on the server, small enough to pass as a prop (about 35 KB of JSON: ports, the
 * official lines, one layer's worth of each field). Pass `data` when the page already called loadData.
 * `region` picks the currents hour with usable coverage there (lib/currents chooseHour). Null without a manifest.
 */
export async function loadStageData({ data, region = null, now = new Date() }: { data?: DataLoad; region?: string | null; now?: Date } = {}): Promise<StageData | null> {
  const d = data ?? (await loadData());
  if (!d.ok) return null;
  const { manifest: m, baseUrl } = d;
  const today = pacificDate(now.toISOString());
  const derived = await derivedIndex(dataBaseUrl());
  const o = derivedOutlook(derived);
  const sst = derivedSst(derived);
  const palette = satelliteLatest(m, "olci300")?.palette ?? satelliteLatest(m, "viirs750")?.palette ?? CHL_PALETTE;
  const hour = chooseHour(m, region).layer;
  return {
    baseUrl,
    ports: d.ports
      ? ({ ...d.ports, features: d.ports.features.map((f) => ({ ...f, properties: { port_code: f.properties.port_code, display_name: f.properties.display_name } })) } as StageData["ports"])
      : null,
    official: slimOfficial(labelledGeometry(d.official)),
    chlorophyll: chlorophyll(m, baseUrl),
    charm: charm(m, baseUrl, today),
    outlook: o
      ? { raster: { id: `${o.id}@${o.issued}`, grid: o.grid, baseUrl, palette }, caption: { what: "Chlorophyll outlook", kind: "outlook", date: `issued ${shortDate(o.issued)}`, ramp: { css: gradientCss(displayPalette(palette)), labels: CHL_TICKS, unit: "mg/m³" } } }
      : null,
    sst: sst
      ? {
          raster: { id: `${sst.id}@${sst.observed_dates[sst.observed_dates.length - 1]}`, grid: sst.grid, baseUrl, palette: SST_PALETTE, style: "observed" },
          caption: { what: "Sea temperature", kind: sstIsAnalysis(sst) ? "analysis" : "observed", date: shortDate(sst.observed_dates[sst.observed_dates.length - 1]), ramp: { css: gradientCss(SST_PALETTE), labels: ["cold", "warm"] } },
        }
      : null,
    currents: hour?.vectors ? { layer: { layer_id: hour.layer_id, vectors: hour.vectors }, hour: `${shortDate(hour.time.valid_time ?? "")}, ${hour.time.valid_time?.slice(11, 16)} UTC` } : null,
  };
}
