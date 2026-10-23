"use client";

import { useEffect, useMemo, useState } from "react";
import type { LayerArtifact, Manifest, ValueGrid } from "@/generated/schema";
import { CHL_NEAR_COPY } from "@/content/copy";
import { pixelStatus } from "@/lib/chlNear";
import { KNOTS_PER_MS, compass, currentsHourly, currentsMean, currentsStale } from "@/lib/currents";
import { artifactUrl, charmLayer, nativeLabel, satelliteDays, satelliteLatest, SAT_PRODUCTS, type SatProduct } from "@/lib/layers";
import { loadGrid, distanceKm } from "@/lib/grid";
import { multiSensorMembers, ratioPhrase } from "@/lib/multisensor";
import { fmtChl } from "@/lib/readout";
import { sensorLabel } from "@/lib/stamp";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel, relTime } from "@/v4/ui/time";
import { useWidth } from "@/v4/charts/useWidth";
import {
  footprintOf,
  loadConsistency,
  loadRadarSampler,
  loadSatSampler,
  obsPoint,
  oceanCoverage,
  radarSeries,
  satDayStrip,
  summarise,
  type BBox,
  type Consistency,
  type ObsPixel,
  type ObsTransectCell,
  type RadarChoice,
  type SatChoice,
  type Sampler,
} from "./obsGrid";
import { download, obsAreaCsv, obsPointCsv, obsTransectCsv } from "./exports";

/**
 * Research tools over observation grids (satellite chlorophyll and HF-radar currents): the exact
 * published pixel at a point, every pixel a transect crosses, every pixel in a box. Each value keeps
 * its own date or hour, sensor and native size; empty pixels are counted and labelled "no
 * observation", never filled, never joined across.
 */

export type ObsSource = "charm" | "satellite" | "radar";

export const SAT_NONE = "No observation — not low chlorophyll. Cloud, fog, sun glint, land or no overpass left this pixel empty; the published grid does not say which, and nothing is filled in.";
export const RADAR_NONE = "No observation — not calm water. No radar coverage here at this hour (or the cell was dropped by the pipeline's QC); nothing is filled in.";

const BTN = "nx-press nx-focus flex min-h-11 items-center gap-2 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold";
const CHIP_ON = "bg-n-accent-bg text-n-accent ring-1 ring-n-accent";
const CHIP_OFF = "border border-n-line-strong text-n-ink-2";

const PRODUCT_LABEL: Record<SatProduct, string> = {
  multi: "Multi-sensor",
  olci300: "Sentinel-3 OLCI 300 m",
  viirs750: "VIIRS 750 m",
};

function ExportButton({ onClick, testid, children }: { onClick: () => void; testid: string; children: React.ReactNode }) {
  return (
    <button type="button" onClick={onClick} className={`${BTN} self-start`} data-testid={testid}>
      <Icon name="download" className="h-4 w-4" /> {children}
    </button>
  );
}

const speedLabel = (v: number) => `${v.toFixed(2)} m/s (${(v * KNOTS_PER_MS).toFixed(1)} kn)`;
export const isMean = (l: LayerArtifact | null | undefined) => !!l?.layer_id.endsWith("mean24h");

/** When an observation layer is for, in words. */
export function obsWhen(l: LayerArtifact | null): string {
  if (!l) return "—";
  if (l.group_id === "currents") return isMean(l) ? `24-hour mean ending ${clockLabel(l.time.valid_time ?? "")}` : `observed ${clockLabel(l.time.valid_time ?? "")}`;
  const c = l.composite;
  if (c) return `latest clear view, pixels observed ${dayLabel(c.oldest_observed_date)}–${dayLabel(c.newest_observed_date)}`;
  if (l.multisensor) return `latest clear view, each pixel dated, newest ${dayLabel(l.time.observed_date ?? "")}`;
  return `overpass ${dayLabel(l.time.observed_date ?? "", true)}`;
}

/** Sensor and native size of a layer, e.g. "Sentinel-3 300 m" or "HF radar, native 2 km". */
export function obsSensor(l: LayerArtifact | null): string {
  if (!l) return "—";
  if (l.group_id === "currents") return `HF radar (HFRNet), native ${nativeLabel(l) ?? "2 km"}`;
  if (l.multisensor) return l.multisensor.members.map((m) => m.label).join(" + ");
  return sensorLabel(l);
}

// ---------------------------------------------------------------- loading

export type ObsRead = {
  status: "idle" | "loading" | "ready" | "failed";
  sampler: Sampler | null;
  error?: string;
};

/**
 * The sampler for an observation layer: radar hours are whole small grids (loaded once per hour);
 * satellite grids are chunked, so only the chunks under `bbox` are fetched.
 */
export function useObsSampler(manifest: Manifest, baseUrl: string, layer: LayerArtifact | null, bbox: BBox | null): ObsRead {
  const key = layer ? `${layer.layer_id}|${layer.group_id === "currents" ? "" : (bbox ?? []).map((x) => x.toFixed(4)).join(",")}` : "";
  const [st, setSt] = useState<{ key: string; read: ObsRead } | null>(null);
  useEffect(() => {
    if (!layer || (layer.group_id !== "currents" && !bbox)) return;
    let live = true;
    const job = layer.group_id === "currents" ? loadRadarSampler(baseUrl, layer) : loadSatSampler(manifest, baseUrl, layer, bbox!);
    job
      .then(
        (s) =>
          live &&
          setSt({
            key,
            read: s
              ? { status: "ready", sampler: s }
              : {
                  status: "failed",
                  sampler: null,
                  error: "No value grid is published for this layer.",
                },
          }),
      )
      .catch(
        (e) =>
          live &&
          setSt({
            key,
            read: {
              status: "failed",
              sampler: null,
              error: (e as Error).message,
            },
          }),
      );
    return () => {
      live = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!layer || (layer.group_id !== "currents" && !bbox)) return { status: "idle", sampler: null };
  return st?.key === key ? st.read : { status: "loading", sampler: null };
}

// ---------------------------------------------------------------- controls

/** Product and published day (satellite), or published hour and the 24-hour mean (radar). Only what was published. */
export function ObsControls({ manifest, source, sat, onSat, radar, onRadar, now }: { manifest: Manifest; source: ObsSource; sat: SatChoice; onSat: (c: SatChoice) => void; radar: RadarChoice; onRadar: (c: RadarChoice) => void; now: Date | null }) {
  if (source === "satellite") {
    const products = SAT_PRODUCTS.filter((p) => satelliteLatest(manifest, p));
    const latest = satelliteLatest(manifest, sat.product);
    const days = satelliteDays(manifest, sat.product);
    return (
      <>
        <fieldset className="min-w-0">
          <legend className="mb-1 text-[12.5px] text-n-ink-3">Sensor</legend>
          <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Satellite sensor">
            {products.map((p) => (
              <button
                key={p}
                type="button"
                role="radio"
                aria-checked={sat.product === p}
                onClick={() => onSat({ product: p, day: null })}
                className={`nx-press nx-focus min-h-11 rounded-lg px-3 text-[14px] font-semibold ${sat.product === p ? CHIP_ON : CHIP_OFF}`}
                data-testid={`sat-product-${p}`}
              >
                {PRODUCT_LABEL[p]}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset className="min-w-0">
          <legend className="mb-1 text-[12.5px] text-n-ink-3">
            Published days
            {days.length ? "" : ` (${PRODUCT_LABEL[sat.product]}: only the latest clear view is published)`}
          </legend>
          <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Satellite day">
            {latest && (
              <button
                type="button"
                role="radio"
                aria-checked={!sat.day}
                onClick={() => onSat({ ...sat, day: null })}
                className={`nx-press nx-focus flex min-h-11 flex-col items-center justify-center rounded-lg px-2.5 leading-tight ${!sat.day ? CHIP_ON : CHIP_OFF}`}
                data-testid="sat-day-latest"
              >
                <span className="text-[13.5px] font-semibold">Latest clear view</span>
                <span className="text-[11.5px]">{latest.composite ? `${dayLabel(latest.composite.oldest_observed_date)}–${dayLabel(latest.composite.newest_observed_date)}` : `to ${dayLabel(latest.time.observed_date ?? "")}`}</span>
              </button>
            )}
            {days.map((d) => {
              const date = d.time.observed_date ?? "";
              const ok = !!d.grid;
              return (
                <button
                  key={d.layer_id}
                  type="button"
                  role="radio"
                  aria-checked={sat.day === date}
                  disabled={!ok}
                  onClick={() => onSat({ ...sat, day: date })}
                  className={`nx-press nx-focus flex min-h-11 flex-col items-center justify-center rounded-lg px-2.5 leading-tight disabled:cursor-not-allowed disabled:border-dashed disabled:opacity-70 ${sat.day === date ? CHIP_ON : CHIP_OFF}`}
                  data-testid={`sat-day-${date}`}
                  title={ok ? undefined : "Published with no clear pixel (cloud or no overpass): no observation that day"}
                >
                  <span className="text-[13.5px] font-semibold">{dayLabel(date, true)}</span>
                  <span className="text-[11.5px]">{ok ? "one overpass" : "no clear pixel"}</span>
                </button>
              );
            })}
          </div>
        </fieldset>
      </>
    );
  }
  if (source === "radar") {
    const hours = currentsHourly(manifest);
    const mean = currentsMean(manifest);
    const value = radar.mean ? "mean" : (radar.hour ?? hours[hours.length - 1]?.time.valid_time ?? "");
    const cur = radar.mean ? mean : (hours.find((h) => h.time.valid_time === value) ?? null);
    const i = hours.findIndex((h) => h.time.valid_time === value);
    const step = (d: number) => {
      const h = hours[Math.max(0, Math.min(hours.length - 1, (i < 0 ? hours.length - 1 : i) + d))];
      if (h?.time.valid_time) onRadar({ hour: h.time.valid_time, mean: false });
    };
    return (
      <fieldset className="min-w-0">
        <legend className="mb-1 text-[12.5px] text-n-ink-3">
          Published radar hour ({hours.length} hours
          {mean ? " and their 24-hour mean" : ""})
        </legend>
        <div className="flex flex-wrap items-center gap-1.5">
          <div className="flex w-full min-w-0 items-center gap-1.5 sm:w-auto">
            <button type="button" onClick={() => step(-1)} disabled={radar.mean || i <= 0} className={`${BTN} px-2.5 disabled:opacity-50`} aria-label="Previous published hour" data-testid="radar-prev">
              <Icon name="back" className="h-4 w-4" />
            </button>
            <label htmlFor="r-radar-hour" className="sr-only">
              Radar hour
            </label>
            <select
              id="r-radar-hour"
              value={value}
              onChange={(e) => onRadar(e.target.value === "mean" ? { hour: null, mean: true } : { hour: e.target.value, mean: false })}
              className="nx-select nx-focus min-h-11 min-w-0 flex-1 rounded-lg border border-n-line-strong bg-n-surface pl-2 text-[14px] font-semibold"
              data-testid="radar-hour"
            >
              {hours.map((h) => (
                <option key={h.layer_id} value={h.time.valid_time ?? ""}>
                  {clockLabel(h.time.valid_time ?? "")}
                </option>
              ))}
              {mean && <option value="mean">24-hour mean, ending {clockLabel(mean.time.valid_time ?? "")}</option>}
            </select>
            <button type="button" onClick={() => step(1)} disabled={radar.mean || i < 0 || i >= hours.length - 1} className={`${BTN} px-2.5 disabled:opacity-50`} aria-label="Next published hour" data-testid="radar-next">
              <Icon name="back" className="h-4 w-4 rotate-180" />
            </button>
          </div>
          {cur && now && currentsStale(cur.time.valid_time, now) && (
            <span className="rounded-md bg-n-surface-2 px-2 py-1 text-[12.5px] font-semibold text-n-stale" data-testid="radar-stale">
              Stale: {relTime(cur.time.valid_time ?? "", now)}
            </span>
          )}
        </div>
      </fieldset>
    );
  }
  return null;
}

// ---------------------------------------------------------------- shared bits

function Quality({ layers, testid = "obs-quality" }: { layers: LayerArtifact[]; testid?: string }) {
  const failed = layers.flatMap((l) => (l.qc?.checks ?? []).filter((c) => !c.passed).map((c) => ({ l, c })));
  const n = layers.reduce((a, l) => a + (l.qc?.checks.length ?? 0), 0);
  return (
    <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid={testid}>
      <b className="font-semibold text-n-ink-2">Layer QC:</b>{" "}
      {failed.length ? (
        <>
          {failed.length} of {n} checks failed:{" "}
          {failed.map(({ l, c }, i) => (
            <span key={`${l.layer_id}-${c.name}-${i}`} className="text-n-down">
              {c.name.replace(/^[^:]*:/, "")} ({c.detail}){i < failed.length - 1 ? "; " : ""}
            </span>
          ))}
          . Values are kept as published; see Lineage below.
        </>
      ) : n ? (
        `all ${n} checks passed.`
      ) : (
        "no checks published."
      )}
    </p>
  );
}

/** Source line: sensor, native size, product id and dataset. */
function SourceLine({ layers }: { layers: LayerArtifact[] }) {
  return (
    <div className="text-[12.5px] leading-snug text-n-ink-3" data-testid="obs-source">
      <p>
        {layers.map((l, i) => (
          <span key={l.layer_id}>
            {i ? "; " : ""}
            <b className="font-semibold text-n-ink-2">{obsSensor(l)}</b>
            {l.group_id !== "currents" && nativeLabel(l) && !l.multisensor ? `, native ${nativeLabel(l)}` : ""} · product <span className="nx-data break-all">{l.layer_id}</span>
          </span>
        ))}
      </p>
      <details>
        <summary className="nx-focus min-h-11 cursor-pointer py-2 font-semibold text-n-accent">Source datasets</summary>
        <ul className="flex flex-col gap-1">
          {layers.map((l) => (
            <li key={l.layer_id} className="nx-data break-all">
              {l.provenance.dataset_id ?? "—"} ({l.provenance.institution ?? "—"})
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

/** Layers whose QC applies: the members, and the multi-sensor display's own checks. */
const qcLayers = (s: Sampler) => (s.layer.multisensor ? [s.layer, ...s.sources] : s.sources);

function sensorOf(s: Sampler, k: 1 | 2 | null): LayerArtifact | null {
  return k ? (s.sources[k - 1] ?? null) : s.sources.length === 1 ? s.sources[0] : null;
}

// ---------------------------------------------------------------- point

export function ObsPointPanel({ manifest, baseUrl, read, px, point, sat, now }: { manifest: Manifest; baseUrl: string; read: ObsRead; px: ObsPixel | null; point: { lat: number; lon: number }; sat: SatChoice; now: Date | null }) {
  const s = read.sampler;
  const radar = s?.kind === "radar";
  const src = s && px ? sensorOf(s, px.sensor) : null;
  const pkey = `${point.lat},${point.lon}`;
  // place rules applied to this pixel (land / shoreline / isolated spike), as in Explore
  const [flag, setFlag] = useState<{ k: string; v: string | null } | null>(null);
  const flagKey = `${pkey}|${src?.layer_id ?? ""}`;
  useEffect(() => {
    if (!src || radar || px?.value == null) return;
    let live = true;
    pixelStatus(baseUrl, src, point.lat, point.lon)
      .then((w) => live && setFlag({ k: flagKey, v: w ? CHL_NEAR_COPY.pixel[w] : null }))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [flagKey]); // eslint-disable-line react-hooks/exhaustive-deps
  // the same pixel in every other published layer: OLCI single days, or every radar hour
  type Other = {
    label: string;
    date: string;
    px: Partial<ObsPixel> | null;
    published: boolean;
    mean?: boolean;
  };
  const [others, setOthers] = useState<{ k: string; rows: Other[] } | null>(null);
  const othersKey = `${pkey}|${s?.kind ?? ""}|${s?.kind === "satellite" ? sat.product : ""}`;
  useEffect(() => {
    if (!s) return;
    let live = true;
    const job: Promise<Other[]> = radar
      ? radarSeries(manifest, baseUrl, point.lat, point.lon).then((r) =>
          r.map((x) => ({
            label: x.layer.layer_id,
            date: x.time,
            px: x.px,
            published: true,
            mean: x.mean,
          })),
        )
      : sat.product === "multi"
        ? Promise.resolve([])
        : satDayStrip(manifest, baseUrl, sat.product, point.lat, point.lon).then((r) =>
            r.map((x) => ({
              label: x.layer.layer_id,
              date: x.date,
              px: x.value == null ? null : { value: x.value, date: x.date },
              published: x.published,
            })),
          );
    job.then((rows) => live && setOthers({ k: othersKey, rows })).catch(() => undefined);
    return () => {
      live = false;
    };
  }, [othersKey, !!s]); // eslint-disable-line react-hooks/exhaustive-deps
  const rows = others?.k === othersKey ? others.rows : null;
  // multi-sensor: both members at this point, with their own dates
  const [members, setMembers] = useState<{
    k: string;
    m: { layer: LayerArtifact; px: ObsPixel | null }[];
  } | null>(null);
  const memKey = `${pkey}|${s?.layer.multisensor ? s.layer.layer_id : ""}`;
  useEffect(() => {
    if (!s?.layer.multisensor) return;
    let live = true;
    const box: BBox = [point.lon - 0.01, point.lat - 0.01, point.lon + 0.01, point.lat + 0.01];
    Promise.all(
      multiSensorMembers(manifest, s.layer).map(async (l) => {
        if (!l) return null;
        const ms = await loadSatSampler(manifest, baseUrl, l, box);
        return { layer: l, px: ms ? obsPoint(ms, point.lat, point.lon) : null };
      }),
    )
      .then(
        (m) =>
          live &&
          setMembers({
            k: memKey,
            m: m.filter((x): x is NonNullable<typeof x> => !!x),
          }),
      )
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [memKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const mem = members?.k === memKey ? members.m : null;

  if (read.status === "failed") return <p className="text-[14px] text-n-down">The grid could not be loaded: {read.error}</p>;
  if (!s) return <p className="text-[14px] text-n-ink-2">Reading the published grid…</p>;
  const res = nativeLabel(src ?? s.sources[0]);
  const fp = px ? footprintOf(s, px) : null;
  return (
    <div className="flex flex-col gap-3" data-testid="obs-point">
      <p className="nx-caps">{radar ? "HF-radar cell" : "Satellite pixel"} · observation, not a forecast</p>
      {!px ? (
        <p className="text-[14px] text-n-ink-2" data-testid="obs-point-none">
          Outside the {radar ? "radar grid" : "satellite grid"}: nothing was observed here by this product.
        </p>
      ) : px.value == null ? (
        <p className="rounded-lg border border-dashed border-n-line-strong px-3 py-2 text-[14px] leading-snug" data-testid="obs-point-none">
          <b className="font-semibold">{radar ? "No observation — not calm water." : "No observation — not low chlorophyll."}</b> {(radar ? RADAR_NONE : SAT_NONE).replace(/^No observation — not [a-z ]+\. /, "")}
        </p>
      ) : (
        <div>
          <p className="nx-data text-[30px] font-semibold leading-none" data-testid="obs-point-value">
            {radar ? px.value.toFixed(2) : fmtChl(px.value)} <span className="text-[14px] font-normal text-n-ink-2">{radar ? "m/s" : "mg/m³ chlorophyll-a"}</span>
          </p>
          {radar && px.dir != null && (
            <p className="mt-1 text-[14px]">
              toward {compass(px.dir)} ({Math.round(px.dir)}°) · {(px.value * KNOTS_PER_MS).toFixed(1)} kn
            </p>
          )}
          <p className="mt-1 text-[14px] font-semibold" data-testid="obs-point-date">
            {radar ? (isMean(s.layer) ? `24-hour mean ending ${clockLabel(px.date ?? "")}` : `Observed ${clockLabel(px.date ?? "")}`) : `Observed ${dayLabel(px.date ?? "", true)}`}
            {now && px.date ? <span className="font-normal text-n-ink-2"> ({relTime(px.date, now)})</span> : null}
          </p>
          {s.layer.composite && px.date && (
            <p className="text-[12.5px] text-n-ink-3">
              This pixel&rsquo;s own date in the latest clear view (window {s.layer.composite.window_days} days to {dayLabel(s.layer.composite.reference_date)}); neighbouring pixels can be from other days.
            </p>
          )}
          {s.layer.multisensor && (
            <p className="mt-1 text-[13.5px]" data-testid="obs-point-sensor">
              Shown by the multi-sensor rule: <b className="font-semibold">{px.sensor ? obsSensor(sensorOf(s, px.sensor)) : "—"}</b>
            </p>
          )}
          {flag?.k === flagKey && flag.v && (
            <p className="mt-1 flex gap-1.5 text-[13px] text-n-stale" data-testid="obs-point-flag">
              <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0" /> {flag.v}
            </p>
          )}
        </div>
      )}
      {px && (
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-[13px]" data-testid="obs-point-meta">
          <dt className="text-n-ink-3">{radar ? "Cell" : "Pixel"}</dt>
          <dd className="nx-data">
            row {px.row}, col {px.col} · centre {px.lat.toFixed(4)}°N {Math.abs(px.lon).toFixed(4)}°W
          </dd>
          {fp && (fp.row !== px.row || fp.col !== px.col) && (
            <>
              <dt className="text-n-ink-3">Native pixel</dt>
              <dd className="nx-data">
                VIIRS row {fp.row}, col {fp.col} · centre {fp.lat.toFixed(4)}°N {Math.abs(fp.lon).toFixed(4)}°W
              </dd>
            </>
          )}
          <dt className="text-n-ink-3">Footprint</dt>
          <dd>
            {px.value == null ? `${res ?? "—"} square, no observation` : `${res ?? "—"} square, outlined on the map`}
            {s.layer.multisensor && px.sensor === 2 ? " (the VIIRS pixel; the display shows it on 300 m pixels)" : ""}
          </dd>
          <dt className="text-n-ink-3">Distance</dt>
          <dd>
            {(distanceKm(point.lat, point.lon, fp!.lat, fp!.lon) * 1000).toFixed(0)} m from the clicked point to the {radar ? "cell" : "pixel"}&rsquo;s centre; the point lies inside it
          </dd>
        </dl>
      )}
      {mem && (
        <section className="flex flex-col gap-1 rounded-lg bg-n-surface-2 p-3 text-[13.5px]" data-testid="obs-point-members">
          <h3 className="nx-caps">Each sensor at this point</h3>
          {mem.map((m) => (
            <p key={m.layer.layer_id}>
              <b className="font-semibold">{obsSensor(m.layer)}:</b> {m.px?.value != null ? `${fmtChl(m.px.value)} mg/m³, observed ${dayLabel(m.px.date ?? "", true)}` : "no observation in its latest clear view"}
            </p>
          ))}
          {mem.length === 2 && mem[0].px?.value != null && mem[1].px?.value != null && mem[0].px.date === mem[1].px.date && (
            <p className="text-[12.5px] text-n-ink-3">Both observed this place on the same day: one pixel pair, not a comparison of the sensors. See the area tool for the consistency check.</p>
          )}
        </section>
      )}
      {rows && rows.length > 0 && (radar ? <RadarStrip rows={rows} sel={s.layer} /> : <DayStrip rows={rows} sel={s.layer} />)}
      {!rows && !s.layer.multisensor && (
        <p className="text-[12.5px] text-n-ink-3">
          Reading the other published {radar ? "hours" : "days"} at this {radar ? "cell" : "pixel"}…
        </p>
      )}
      {!radar && sat.product === "viirs750" && <p className="text-[12.5px] text-n-ink-3">VIIRS single-day grids are not published; only its latest clear view, with each pixel dated.</p>}
      <SourceLine layers={src ? [src] : s.sources} />
      <Quality layers={src ? [src] : qcLayers(s)} />
      {px && (
        <ExportButton
          testid="export-obs-point"
          onClick={() =>
            download(
              `neritic-${s.layer.layer_id}-pixel-${px.row}-${px.col}.csv`,
              obsPointCsv(
                s,
                px,
                (rows ?? []).map((r) => ({
                  label: r.label,
                  px: r.published ? r.px : null,
                })),
                manifest,
              ),
            )
          }
        >
          This {radar ? "cell" : "pixel"}
          {rows?.length ? `, every published ${radar ? "hour" : "day"}` : ""} (CSV)
        </ExportButton>
      )}
    </div>
  );
}

function DayStrip({
  rows,
  sel,
}: {
  rows: {
    label: string;
    date: string;
    px: Partial<ObsPixel> | null;
    published: boolean;
  }[];
  sel: LayerArtifact;
}) {
  return (
    <section className="flex flex-col gap-1" data-testid="obs-day-strip">
      <h3 className="nx-caps">This pixel on each published day</h3>
      <ul className="grid grid-cols-[repeat(auto-fill,minmax(88px,1fr))] gap-1">
        {rows.map((r) => (
          <li key={r.label} className={`rounded-md px-2 py-1.5 text-center leading-tight ${r.label === sel.layer_id ? "ring-1 ring-n-accent" : ""} ${r.px?.value != null ? "bg-n-surface-2" : "border border-dashed border-n-line-strong"}`} data-testid="obs-day">
            <span className="block text-[12px] text-n-ink-3">{dayLabel(r.date, true)}</span>
            <span className="nx-data block text-[14px] font-semibold">{r.px?.value != null ? fmtChl(r.px.value) : "—"}</span>
            <span className="block text-[11px] text-n-ink-3">{r.px?.value != null ? "mg/m³" : r.published ? "no observation" : "no clear pixel"}</span>
          </li>
        ))}
      </ul>
      <p className="text-[12.5px] text-n-ink-3">A dash is no observation that day (cloud or no overpass), not low chlorophyll. Days are separate overpasses; nothing is drawn between them.</p>
    </section>
  );
}

function RadarStrip({
  rows,
  sel,
}: {
  rows: {
    label: string;
    date: string;
    px: Partial<ObsPixel> | null;
    mean?: boolean;
  }[];
  sel: LayerArtifact;
}) {
  const [box, W] = useWidth<HTMLElement>(340);
  const hours = rows.filter((r) => !r.mean);
  const mean = rows.find((r) => r.mean);
  const top = Math.max(0.5, ...hours.map((h) => h.px?.value ?? 0));
  const H = 96;
  const pad = { l: 34, r: 6, t: 6, b: 18 };
  const bw = (W - pad.l - pad.r) / Math.max(1, hours.length);
  const y = (v: number) => pad.t + (1 - v / top) * (H - pad.t - pad.b);
  const n = hours.filter((h) => h.px?.value != null).length;
  return (
    <figure ref={box} className="flex flex-col gap-1" data-testid="obs-radar-strip">
      <figcaption className="nx-caps">This cell, each published hour</figcaption>
      <svg width={W} height={H} className="block" role="img" aria-label={`Current speed at this cell: ${n} of ${hours.length} hours observed`}>
        <defs>
          <pattern id="nx-hatch-r" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--nx-line-strong)" strokeWidth="1.5" />
          </pattern>
        </defs>
        {[0, top].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--nx-line)" />
            <text x={pad.l - 4} y={y(v) + 4} textAnchor="end" fontSize="10.5" fill="var(--nx-ink-3)">
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        {hours.map((h, i) => {
          const x = pad.l + i * bw;
          const on = h.label === sel.layer_id;
          return h.px?.value == null ? (
            <rect key={h.label} x={x + 0.5} y={pad.t} width={Math.max(1, bw - 1)} height={H - pad.t - pad.b} fill="url(#nx-hatch-r)" />
          ) : (
            <rect key={h.label} x={x + 0.5} y={y(h.px.value)} width={Math.max(1, bw - 1)} height={y(0) - y(h.px.value)} fill="var(--nx-conditions)" opacity={on ? 1 : 0.55} />
          );
        })}
        <text x={pad.l} y={H - 4} fontSize="10.5" fill="var(--nx-ink-3)">
          {hours[0] ? clockLabel(hours[0].date).replace(/ [A-Z]{3}$/, "") : ""}
        </text>
        <text x={W - pad.r} y={H - 4} fontSize="10.5" fill="var(--nx-ink-3)" textAnchor="end">
          {hours.at(-1) ? clockLabel(hours.at(-1)!.date) : ""}
        </text>
      </svg>
      <p className="text-[12.5px] text-n-ink-3">
        Speed in m/s; {n} of {hours.length} hours observed here. Hatched: no radar observation that hour (no data, not calm water); bars are separate hours, never joined.
        {mean ? ` 24-hour mean here: ${mean.px?.value != null ? `${speedLabel(mean.px.value)} toward ${compass(mean.px.dir ?? 0)}` : "none published (fewer than 18 of 24 hours observed)"}.` : ""}
      </p>
    </figure>
  );
}

// ---------------------------------------------------------------- transect

export function ObsTransect({ manifest, read, cells, a, b }: { manifest: Manifest; read: ObsRead; cells: ObsTransectCell[] | null; a: { lat: number; lon: number }; b: { lat: number; lon: number } }) {
  const s = read.sampler;
  if (read.status === "failed") return <p className="text-[14px] text-n-down">The grid could not be loaded: {read.error}</p>;
  if (!s || !cells) return <p className="text-[14px] text-n-ink-2">Reading the published grid…</p>;
  const sum = summarise(cells);
  const radar = s.kind === "radar";
  return (
    <div className="flex flex-col gap-2" data-testid="obs-transect">
      <ObsTransectChart cells={cells} a={a} b={b} radar={radar} multi={!!s.layer.multisensor} />
      <p className="text-[13.5px] leading-snug" data-testid="obs-transect-counts">
        <span className="nx-data font-semibold">{sum.observed}</span> of {sum.inGrid} {radar ? "radar cells" : "pixels"} crossed have an observation; {sum.inGrid - sum.observed} have none
        {radar ? " (no data, not calm water)" : " (not low chlorophyll)"}
        {cells.some((c) => !c.inDomain) ? "; part of the line lies outside the product's grid" : ""}.
        {sum.dates.length > 0 && (
          <>
            {" "}
            {radar ? "Hour" : "Observed"}: {sum.dates.map((d) => `${radar ? clockLabel(d.key) : dayLabel(d.key)} (${d.n})`).join(", ")}.
          </>
        )}
      </p>
      <SourceLine layers={s.sources} />
      <Quality layers={qcLayers(s)} />
      <ExportButton testid="export-obs-transect" onClick={() => download(`neritic-transect-${s.layer.layer_id}.csv`, obsTransectCsv(s, a, b, cells, manifest))}>
        Transect {radar ? "cells" : "pixels"} (CSV)
      </ExportButton>
    </div>
  );
}

const LOG_LO = -2;

function ObsTransectChart({ cells, a, b, radar, multi }: { cells: ObsTransectCell[]; a: { lat: number; lon: number }; b: { lat: number; lon: number }; radar: boolean; multi: boolean }) {
  const [box, W] = useWidth<HTMLElement>(640);
  const end = Math.max(0.1, distanceKm(a.lat, a.lon, b.lat, b.lon));
  const H = 168;
  const band = multi ? 8 : 0;
  const pad = { l: 44, r: 8, t: 8, b: 26 + band };
  const vals = cells.flatMap((c) => (c.value == null ? [] : [c.value]));
  // fixed axes: chlorophyll on decades 0.01–100 mg/m³ (to 1000 only when a value needs it); speed 0–1 m/s (2 when needed)
  const hiLog = vals.some((v) => v > 100) ? 3 : 2;
  const top = radar ? (vals.some((v) => v > 1) ? 2 : 1) : hiLog;
  const x = (km: number) => pad.l + (km / end) * (W - pad.l - pad.r);
  const yv = (v: number) => {
    const f = radar ? v / top : (Math.log10(Math.max(v, 10 ** LOG_LO)) - LOG_LO) / (hiLog - LOG_LO);
    return pad.t + (1 - f) * (H - pad.t - pad.b);
  };
  const ticks = radar ? [0, top / 2, top] : Array.from({ length: hiLog - LOG_LO + 1 }, (_, i) => 10 ** (LOG_LO + i));
  const ll = (p: { lat: number; lon: number }) => `${p.lat.toFixed(3)}°N ${Math.abs(p.lon).toFixed(3)}°W`;
  const valued = vals.length;
  const color = radar ? "var(--nx-conditions)" : "var(--nx-measured)";
  return (
    <figure ref={box} className="flex flex-col gap-1">
      <svg width={W} height={H} className="block" role="img" aria-label={`Transect of ${cells.length} ${radar ? "radar cells" : "pixels"}; ${valued} with an observation`} data-testid="obs-transect-svg">
        <defs>
          <pattern id="nx-hatch-o" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--nx-line-strong)" strokeWidth="1.5" />
          </pattern>
        </defs>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={yv(v)} y2={yv(v)} stroke="var(--nx-line)" />
            <text x={pad.l - 5} y={yv(v) + 4} textAnchor="end" fontSize="11" fill="var(--nx-ink-3)">
              {radar ? v.toFixed(1) : v >= 1 ? v : v.toFixed(2)}
            </text>
          </g>
        ))}
        {cells.map((c, i) => {
          const x0 = x(c.km);
          const x1 = x(cells[i + 1]?.km ?? end);
          return c.value == null ? (
            <rect key={i} x={x0} y={pad.t} width={Math.max(1, x1 - x0)} height={H - pad.t - pad.b} fill="url(#nx-hatch-o)" data-gap="1" />
          ) : (
            <g key={i}>
              <line x1={x0} x2={x1} y1={yv(c.value)} y2={yv(c.value)} stroke={color} strokeWidth={2.2} />
              {multi && <rect x={x0} y={H - pad.b + 4} width={Math.max(1, x1 - x0)} height={band - 2} fill={c.sensor === 1 ? "var(--nx-ink-2)" : "var(--nx-accent)"} opacity={c.sensor === 1 ? 0.9 : 0.55} />}
            </g>
          );
        })}
        <text x={pad.l} y={H - 6} fontSize="11" fill="var(--nx-ink-3)">
          0 km
        </text>
        <text x={W - pad.r} y={H - 6} fontSize="11" fill="var(--nx-ink-3)" textAnchor="end">
          {end.toFixed(1)} km
        </text>
        <text x={4} y={pad.t + 4} fontSize="10.5" fill="var(--nx-ink-3)" transform={`rotate(-90 4 ${pad.t + 4})`} textAnchor="end" dominantBaseline="hanging">
          {radar ? "m/s" : "mg/m³, log"}
        </text>
      </svg>
      <figcaption className="text-[12.5px] leading-snug text-n-ink-3">
        {cells.length} {radar ? "radar cells" : "pixels"} crossed, {valued} with an observation; each value is held flat across its own {radar ? "cell" : "pixel"} and never joined across a gap. Hatched: no observation (
        {radar ? "no data, not calm water" : "not low chlorophyll"}
        ).
        {multi ? " Band under the axis: dark = Sentinel-3 300 m, blue = VIIRS 750 m (the sensor shown at each pixel); a VIIRS value repeats across the 300 m display pixels inside its 750 m pixel." : ""} From {ll(a)} to {ll(b)}.
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------- area

export function ObsArea({ manifest, baseUrl, read, cells, bbox, now }: { manifest: Manifest; baseUrl: string; read: ObsRead; cells: ObsPixel[] | null; bbox: BBox; now: Date | null }) {
  const s = read.sampler;
  // the coverage reference the pipeline uses: C-HARM ocean cells (Pseudo-nitzschia nowcast has a value)
  const ref = charmLayer(manifest, "pseudo_nitzschia", 0);
  const [refCodes, setRefCodes] = useState<{
    id: string;
    codes: Uint16Array;
  } | null>(null);
  useEffect(() => {
    if (!ref?.grid || refCodes?.id === ref.layer_id) return;
    let live = true;
    loadGrid(artifactUrl(baseUrl, ref.grid.url))
      .then((c) => live && setRefCodes({ id: ref.layer_id, codes: c }))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [ref, baseUrl, refCodes?.id]);
  const sum = useMemo(() => (cells ? summarise(cells) : null), [cells]);
  const cov = useMemo(() => (s && ref?.grid && refCodes?.id === ref.layer_id ? oceanCoverage({ g: ref.grid as ValueGrid, codes: refCodes.codes }, bbox, s) : null), [s, ref, refCodes, bbox]);
  if (read.status === "failed") return <p className="text-[14px] text-n-down">The grid could not be loaded: {read.error}</p>;
  if (!s || !sum || !cells) return <p className="text-[14px] text-n-ink-2">Reading the published grid…</p>;
  const radar = s.kind === "radar";
  const unit = radar ? "m/s" : "mg/m³";
  const fmt = (v: number | null) => (v == null ? "—" : radar ? v.toFixed(2) : fmtChl(v));
  const what = radar ? "radar cells" : s.layer.multisensor ? "display pixels (300 m)" : "pixels";
  return (
    <section className="flex flex-col gap-2 border-t border-n-line pt-4" data-testid="obs-area">
      <h3 className="nx-caps">
        {radar ? "Radar cells" : "Satellite pixels"} in the box · {obsWhen(s.layer)}
      </h3>
      <p className="text-[14.5px] leading-snug" data-testid="obs-area-counts">
        <span className="nx-data font-semibold">{sum.observed.toLocaleString("en-US")}</span> of {sum.inGrid.toLocaleString("en-US")} {what} in the box have an observation;{" "}
        <span className="nx-data font-semibold">{(sum.inGrid - sum.observed).toLocaleString("en-US")}</span> have none ({radar ? "no radar data or land: not calm water" : "cloud, fog, land or no overpass: not low chlorophyll"}
        ).
      </p>
      {cov && (
        <p className="text-[13.5px] leading-snug" data-testid="obs-area-coverage">
          Ocean coverage: <span className="nx-data font-semibold">{cov.observed}</span> of {cov.cells} C-HARM ocean cells (about 3 km) in the box hold at least one observed {radar ? "cell" : "pixel"}
          {cov.cells ? ` (${Math.round((cov.observed / cov.cells) * 100)}%)` : ""}, by the pipeline&rsquo;s own coverage rule.
        </p>
      )}
      {sum.observed > 0 && (
        <dl className="grid grid-cols-3 gap-2 text-center">
          {(
            [
              ["Lowest", sum.min],
              ["Median", sum.median],
              ["Highest", sum.max],
            ] as const
          ).map(([k, v]) => (
            <div key={k} className="rounded-lg bg-n-surface-2 py-2">
              <dt className="text-[12px] text-n-ink-3">{k}</dt>
              <dd className="nx-data text-[18px] font-semibold leading-tight">
                {fmt(v)}
                <span className="block text-[11.5px] font-normal text-n-ink-3">{unit}</span>
              </dd>
            </div>
          ))}
        </dl>
      )}
      {sum.dates.length > 0 && !radar && (
        <div className="text-[13px]" data-testid="obs-area-dates">
          <p className="font-semibold">{radar ? "Hour of the observations" : "When each pixel was observed"}</p>
          <ul className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-n-ink-2">
            {sum.dates.map((d) => (
              <li key={d.key}>
                <span className="nx-data">{radar ? (isMean(s.layer) ? `24 h mean ending ${clockLabel(d.key)}` : clockLabel(d.key)) : dayLabel(d.key, true)}</span>: {d.n.toLocaleString("en-US")}
                {now && !radar ? <span className="text-n-ink-3"> ({relTime(d.key, now)})</span> : null}
              </li>
            ))}
          </ul>
          {s.layer.composite && <p className="text-[12.5px] text-n-ink-3">Latest clear view: each pixel is its newest observation within {s.layer.composite.window_days} days; neighbouring pixels can be days apart.</p>}
        </div>
      )}
      {sum.sensors.length > 0 && (
        <p className="text-[13px]" data-testid="obs-area-sensors">
          <b className="font-semibold">Sensor shown:</b>{" "}
          {sum.sensors.map((x, i) => `${i ? "; " : ""}${obsSensor(sensorOf(s, x.key))} ${x.n.toLocaleString("en-US")} pixels${x.key === 2 ? ` (from ${sum.nativeSecondary.toLocaleString("en-US")} native 750 m VIIRS pixels)` : ""}`).join("")}.
        </p>
      )}
      <p className="text-[12.5px] leading-snug text-n-ink-3">A count and spread of the published {radar ? "cells" : "pixels"} in the box: not an average for any harbour. Gaps are counted, never filled.</p>
      <SourceLine layers={s.sources} />
      <Quality layers={qcLayers(s)} />
      <ExportButton testid="export-obs-area" onClick={() => download(`neritic-area-${s.layer.layer_id}.csv`, obsAreaCsv(s, bbox, cells, manifest))}>
        {radar ? "Cells" : "Pixels"} in the box (CSV)
      </ExportButton>
      {s.layer.multisensor && <Consistency manifest={manifest} baseUrl={baseUrl} layer={s.layer} bbox={bbox} />}
    </section>
  );
}

// ---------------------------------------------------------------- inter-sensor consistency (multi-sensor)

function Consistency({ manifest, baseUrl, layer, bbox }: { manifest: Manifest; baseUrl: string; layer: LayerArtifact; bbox: BBox }) {
  const ms = layer.multisensor!;
  const key = `${layer.layer_id}|${bbox.map((x) => x.toFixed(4)).join(",")}`;
  const [st, setSt] = useState<{
    k: string;
    c: Consistency | null;
    err?: string;
  } | null>(null);
  useEffect(() => {
    let live = true;
    loadConsistency(manifest, baseUrl, layer, bbox)
      .then((c) => live && setSt({ k: key, c: c ? { ...c } : null }))
      .catch((e) => live && setSt({ k: key, c: null, err: (e as Error).message }));
    return () => {
      live = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  const c = st?.k === key ? st.c : undefined;
  const published = ms.agreement.filter((a) => a.n_cells > 0 && a.median_log10_ratio != null);
  const cov = ms.coverage_comparison.find((x) => x.region_id === "domain");
  return (
    <section className="mt-2 flex flex-col gap-2 rounded-lg border border-n-line p-3" data-testid="obs-consistency">
      <h3 className="nx-caps">Inter-sensor consistency check · Sentinel-3 and VIIRS</h3>
      <p className="text-[12.5px] leading-snug text-n-ink-3">
        How the two sensors compare where both saw the same water on the same day. A consistency check between two sensors, not a validation: it does not say which sensor is right, and neither is compared with water samples.
      </p>
      {c === undefined ? (
        st?.err ? (
          <p className="text-[13.5px] text-n-down">The member grids could not be loaded: {st.err}</p>
        ) : (
          <p className="text-[13.5px] text-n-ink-2">Reading both sensors&rsquo; grids…</p>
        )
      ) : !c ? (
        <p className="text-[13.5px] text-n-ink-2">The member layers of this display are not published with value grids, so no pair can be formed.</p>
      ) : c.pairs.length ? (
        <div className="text-[13.5px] leading-snug" data-testid="obs-consistency-result">
          <p>
            <b className="font-semibold">{c.pairs.length}</b> same-day VIIRS cell{c.pairs.length === 1 ? "" : "s"} in this box with at least {c.minPixels} Sentinel-3 pixels of that day inside (
            {[...new Set(c.pairs.map((p) => p.date))].map((d) => dayLabel(d)).join(", ")}
            ): median log10(Sentinel-3 / VIIRS) = <span className="nx-data font-semibold">{c.medianLog10!.toFixed(3)}</span>, i.e. {ratioPhrase(c.medianLog10!)}.
          </p>
          {c.pairs.length < 10 && <p className="text-[12.5px] text-n-stale">Fewer than 10 pairs: too few to say how the sensors compare in general.</p>}
        </div>
      ) : (
        <p className="text-[13.5px] leading-snug" data-testid="obs-consistency-none">
          <b className="font-semibold">Agreement cannot be computed in this box.</b>{" "}
          {c.viirsCells === 0
            ? "VIIRS has no observation here in its latest clear view."
            : c.sameDayCandidates === 0
              ? `VIIRS observed ${c.viirsCells.toLocaleString("en-US")} cells here, on ${c.noS3Days.map((d) => dayLabel(d, true)).join(", ")}; Sentinel-3 published no observation on ${c.noS3Days.length === 1 ? "that day" : "those days"} (no clear pixel or no overpass), so no value of both sensors exists for the same place and date.`
              : `Of ${c.viirsCells.toLocaleString("en-US")} VIIRS cells observed here, ${c.sameDayCandidates.toLocaleString("en-US")} fall on a day Sentinel-3 also published, but none has at least ${c.minPixels} valid Sentinel-3 pixels of that day inside it.`}{" "}
          VIIRS values come from its latest clear view (its single days are not published), so a place VIIRS saw again later cannot be paired for an earlier day.
        </p>
      )}
      <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="obs-consistency-published">
        <b className="font-semibold text-n-ink-2">The pipeline&rsquo;s own figure:</b>{" "}
        {published.length
          ? published.map((a) => `${a.label}: ${ratioPhrase(a.median_log10_ratio!)} (${a.n_cells.toLocaleString("en-US")} cells; ${a.dates.map((d) => dayLabel(d)).join(", ")})`).join("; ")
          : "no same-day pairs this week in any region, so no agreement figure is published."}
        {cov ? ` Published coverage of the whole domain: Sentinel-3 ${Math.round(cov.primary_fraction * 100)}%, VIIRS ${Math.round(cov.secondary_fraction * 100)}%, either ${Math.round(cov.combined_fraction * 100)}% of ocean cells.` : ""}
      </p>
      <details className="text-[12.5px] text-n-ink-2">
        <summary className="nx-focus min-h-11 cursor-pointer py-2 font-semibold text-n-accent">Method, as published</summary>
        <p>&ldquo;{ms.agreement_method}&rdquo;</p>
        <p className="mt-1">
          This box uses the same method on the published grids: Sentinel-3 from its single-day grid of the VIIRS date when published, otherwise from its latest clear view&rsquo;s pixels of that date. The display rule: &ldquo;{ms.rule}&rdquo;
        </p>
      </details>
    </section>
  );
}

/** Explanatory line under the controls: what the chosen observation is and is not. */
export function ObsNote({ source }: { source: ObsSource }) {
  if (source === "satellite")
    return (
      <p className="text-[12.5px] leading-snug text-n-ink-3">
        Observation, not a forecast: each value is one published pixel with its own observation date. Chlorophyll-a is algal biomass, not a toxin and not Pseudo-nitzschia. An empty pixel is no observation, never low chlorophyll; gaps are never filled or
        interpolated.
      </p>
    );
  if (source === "radar")
    return (
      <p className="text-[12.5px] leading-snug text-n-ink-3">
        Observed surface currents (HF radar, 2 km cells), not a forecast and not where a bloom will go. Only published hours; each hour is a separate snapshot. An empty cell is no observation, never calm water; gaps are never filled or interpolated.
      </p>
    );
  return null;
}
