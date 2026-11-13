"use client";

import { useEffect, useId, useMemo, useState } from "react";
import { ChartMarks, type ChartMark } from "@/nx/chart/Labels";
import { latY, lngX, type Bounds } from "@/nx/chart/mercator";
import { ChartRegulation, shoreLon } from "@/nx/chart/regulation";
import { ChartScene } from "@/nx/chart/scene";
import { MapLayers, Micro, usePhone } from "@/nx/ds";
import { day, time } from "@/nx/ds/format";
import { keepTripsOffline } from "@/nx/offline/trips";
import type { Snapshot, Trip } from "../types";
import { tripGeo, type GeoPort, type TripGeo } from "./tripGeo";
import s from "./chart.module.css";

/**
 * The trip on the chart (v15): the port, its whale zone, the CDPH test site and the buoy, drawn where they are, from
 * the trip's snapshot. It writes to the one persistent chart behind the page (ChartScene, ChartRegulation, ChartMarks,
 * as My Coast does), so the workspace keeps the v14 chartroom; offline the static tier's baked chart takes the same
 * marks. Where no chart can show (no chart host, a map that never drew and no baked picture on this device), it draws
 * the same features as a still chart plate in the page, built from the snapshot and the coastline the app ships.
 *
 * Restrained chrome: no route, distance, ground or position claim; the plate says "Not for navigation" as the
 * chart's scale bar does. A record with no position (a test site the snapshot does not place) is not drawn.
 */
export type TripChartProps = {
  trip: Pick<Trip, "id" | "fishery" | "placeKey" | "departKey" | "name">;
  snapshot: Snapshot | null;
  port: GeoPort | null;
  /** "auto": the live chart, with the plate when it cannot run; "plate": only the plate (print); "scene": only the live chart */
  variant?: "auto" | "plate" | "scene";
  className?: string;
};

export function TripChart({ trip, snapshot, port, variant = "auto", className }: TripChartProps) {
  const geo = useMemo(() => (port ? tripGeo(trip, snapshot, port) : null), [trip, snapshot, port]);
  const live = useLiveChart(variant === "auto");
  const drives = variant === "auto" || variant === "scene";
  // a workspace opened online keeps the trips page and its code for offline use (once per page view)
  useEffect(() => {
    if (drives) void keepTripsOffline();
  }, [drives]);
  if (!geo) return variant === "plate" ? <PlateEmpty className={className} /> : null;
  return (
    <>
      {drives ? <LiveChart tripId={trip.id} geo={geo} /> : null}
      {variant === "plate" || live === "fallback" ? <TripPlate geo={geo} className={className} /> : null}
    </>
  );
}

/**
 * Whether the persistent chart is showing the trip: "live" once its map has drawn or its baked picture has loaded (the
 * host shows that picture on the static tier and while a live map is still loading; the trips page keeps the pictures
 * for offline use). "fallback" only when no chart host is mounted, or when neither the map nor a picture has appeared
 * after a generous wait. Once the chart has shown, it stays "live": the plate never draws over a working chart.
 */
export function useLiveChart(enabled = true): "pending" | "live" | "fallback" {
  const [state, setState] = useState<"pending" | "live" | "fallback">("pending");
  useEffect(() => {
    if (!enabled) return;
    const sheet = document.querySelector<HTMLElement>('[data-testid="nx-sheet"]');
    if (!sheet) {
      setState("fallback");
      return;
    }
    let shown = false;
    const showing = () => sheet.dataset.mapReady === "1" || [...sheet.querySelectorAll<HTMLImageElement>('[data-testid="nx-poster"] img')].some((i) => i.complete && i.naturalWidth > 0);
    const read = () => {
      if (shown || !showing()) return;
      shown = true;
      setState("live");
    };
    read();
    const mo = new MutationObserver(read);
    mo.observe(sheet, { attributes: true, attributeFilter: ["data-tier", "data-map-ready"], subtree: true, childList: true });
    const poll = window.setInterval(read, 500);
    const late = window.setTimeout(() => setState((v) => (v === "pending" && !shown ? "fallback" : v)), 8000);
    return () => {
      mo.disconnect();
      window.clearInterval(poll);
      window.clearTimeout(late);
    };
  }, [enabled]);
  return state;
}

function LiveChart({ tripId, geo }: { tripId: string; geo: TripGeo }) {
  const phone = usePhone();
  const { port, zone, buoy, test } = geo;
  const marks: ChartMark[] = [];
  if (buoy && (buoy.waveFt != null || buoy.windKt != null)) {
    const waves = buoy.waveFt != null;
    marks.push({
      id: `trip-buoy-${buoy.id}`,
      lon: buoy.lon,
      lat: buoy.lat,
      layer: "sea",
      priority: 70,
      leader: true,
      offset: phone ? 14 : 18,
      // the page covers the chart's west side on a wide screen: the tag goes east of the buoy first (as My Coast's does)
      anchors: phone ? ["bottom-left", "left", "top-left", "bottom", "top"] : ["bottom-right", "right", "bottom", "bottom-left", "left", "top-left", "top-right"],
      label: `Buoy ${buoy.id}${buoy.at ? `, measured ${day(buoy.at)} ${time(buoy.at)}` : ""}: ${waves ? `waves ${buoy.waveFt!.toFixed(1)} ft${buoy.periodS != null ? ` at ${Math.round(buoy.periodS)} s` : ""}` : `wind ${Math.round(buoy.windKt!)} kt`}, as saved with this trip`,
      content: (
        <div className={s.tag}>
          <Micro glyph="measured">
            Buoy {buoy.id}
            {buoy.at ? ` · ${day(buoy.at)} ${time(buoy.at)}` : ""}
          </Micro>
          <p className={s.tagFig}>
            {waves ? buoy.waveFt!.toFixed(1) : Math.round(buoy.windKt!)} <small>{waves ? `ft${buoy.periodS != null ? ` at ${Math.round(buoy.periodS)} s` : ""}` : `kt${buoy.windDir ? ` ${buoy.windDir}` : ""}`}</small>
          </p>
        </div>
      ),
    });
  }
  if (test) {
    marks.push({
      id: `trip-test-${test.key}`,
      lon: test.lon,
      lat: test.lat,
      layer: "toxin",
      priority: 80,
      leader: true,
      insist: true,
      offset: 18,
      anchors: ["top-left", "top", "left", "top-right", "bottom-left"],
      point: <i className={s.testPoint} />,
      label: `CDPH test, ${test.site}${test.date ? `, ${day(test.date)}` : ""}: highest ${test.max ?? "not detected"}${test.action != null ? ` of ${test.action} ppm` : " ppm"}${test.old ? ", more than 30 days ago" : ""}`,
      content: (
        <div className={s.tag}>
          <Micro glyph={test.old ? "old" : "measured"}>
            CDPH · {test.site}
            {test.date ? ` · ${day(test.date)}` : ""}
          </Micro>
          <p className={s.tagFig}>
            {test.max == null ? "ND" : test.max} <small>{test.action != null ? `of ${test.action} ppm` : "ppm"}</small>
          </p>
        </div>
      ),
    });
  }
  return (
    <>
      <ChartScene scene={{ id: `trips:${tripId}:${port.key}`, view: { bounds: geo.bounds, maxZoom: 11 }, focus: port }} />
      <MapLayers page="trips" defaults={{ regulation: true, depth: true, sea: true, toxin: true }} />
      <ChartRegulation spec={zone ? { zones: { highlight: zone.zone, caption: geo.whale?.text ?? null } } : null} />
      <ChartMarks marks={marks} />
    </>
  );
}

// ---------------------------------------------------------------- the still plate

const W = 360;

/** Web Mercator onto the plate: x east, y south, in plate pixels. */
function projector(b: Bounds, w: number) {
  const x0 = lngX(b[0][0]);
  const x1 = lngX(b[1][0]);
  const y0 = latY(b[1][1]);
  const y1 = latY(b[0][1]);
  const k = w / (x1 - x0);
  return { h: Math.round((y1 - y0) * k), p: (lon: number, lat: number) => [(lngX(lon) - x0) * k, (latY(lat) - y0) * k] as const };
}

/** The mainland between the frame's latitudes, as one closed path (west shore, then the frame's east edge). */
function landPath(b: Bounds, p: (lon: number, lat: number) => readonly [number, number]): string | null {
  const pts: Array<readonly [number, number]> = [];
  const step = 0.02;
  for (let lat = b[0][1]; lat <= b[1][1] + 1e-9; lat += step) {
    const lon = shoreLon(Math.min(lat, b[1][1]));
    if (lon != null) pts.push(p(Math.min(lon, b[1][0] + 0.5), lat));
  }
  if (pts.length < 2) return null;
  const east = p(b[1][0] + 0.5, b[1][1]);
  const southEast = p(b[1][0] + 0.5, b[0][1]);
  const f = (v: number) => v.toFixed(1);
  return `M${pts.map(([x, y]) => `${f(x)} ${f(y)}`).join("L")}L${f(east[0])} ${f(pts[pts.length - 1][1])}L${f(east[0])} ${f(east[1])}L${f(southEast[0])} ${f(southEast[1])}L${f(southEast[0])} ${f(pts[0][1])}Z`;
}

/** Graticule spacing for a frame of this height: about four lines. */
const gridStep = (span: number) => [0.05, 0.1, 0.2, 0.25, 0.5].find((g) => span / g <= 5) ?? 1;
const minutes = (v: number, hemi: [string, string]) => {
  const a = Math.abs(v);
  const d = Math.floor(a + 1e-9);
  const m = Math.round((a - d) * 60);
  return `${d}°${m ? ` ${String(m).padStart(2, "0")}′` : ""} ${v >= 0 ? hemi[0] : hemi[1]}`;
};

export function TripPlate({ geo, className }: { geo: TripGeo; className?: string }) {
  const id = useId();
  const { port, zone, buoy, test } = geo;
  const b = geo.bounds;
  const { h, p } = projector(b, W);
  const land = landPath(b, p);
  const step = gridStep(b[1][1] - b[0][1]);
  const lats: number[] = [];
  for (let v = Math.ceil(b[0][1] / step) * step; v < b[1][1]; v += step) lats.push(Number(v.toFixed(3)));
  const lonStep = step * 2;
  const lons: number[] = [];
  for (let v = Math.ceil(b[0][0] / lonStep) * lonStep; v < b[1][0]; v += lonStep) lons.push(Number(v.toFixed(3)));
  const [px, py] = p(port.lon, port.lat);
  const zoneLines = zone ? [zone.north, zone.south].filter((l) => l > b[0][1] && l < b[1][1]) : [];
  const zTop = zone ? p(b[0][0], Math.min(zone.north, b[1][1]))[1] : 0;
  const zBot = zone ? p(b[0][0], Math.max(zone.south, b[0][1]))[1] : 0;
  const zoneLabelY = zone ? Math.min(Math.max(zTop + 18, 18), h - 30) : 0;
  const title = [
    `Chart of ${port.name}`,
    zone ? `whale zone ${zone.zone}${geo.whale ? ` (${geo.whale.text})` : ""}` : null,
    test ? `CDPH test site ${test.site}` : null,
    buoy ? `buoy ${buoy.id}` : null,
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <figure className={`${s.plate} ${className ?? ""}`} data-testid="trip-chart-plate">
      <svg viewBox={`0 0 ${W} ${h}`} className={s.svg} role="img" aria-labelledby={`${id}-t`}>
        <title id={`${id}-t`}>{`${title}. Not for navigation.`}</title>
        <rect width={W} height={h} className={s.sea} />
        {zone ? <rect x={0} y={Math.max(0, zTop)} width={W} height={Math.max(0, Math.min(h, zBot) - Math.max(0, zTop))} className={s.zoneBand} /> : null}
        {lats.map((v) => {
          const y = p(b[0][0], v)[1];
          return (
            <g key={`la${v}`}>
              <line x1={0} x2={W} y1={y} y2={y} className={s.grat} />
              <text x={6} y={y - 4} className={s.gratText}>
                {minutes(v, ["N", "S"])}
              </text>
            </g>
          );
        })}
        {lons.map((v) => {
          const x = p(v, b[0][1])[0];
          return <line key={`lo${v}`} x1={x} x2={x} y1={0} y2={h} className={s.grat} />;
        })}
        {land ? <path d={land} className={s.land} /> : null}
        {zoneLines.map((l) => {
          const y = p(b[0][0], l)[1];
          return <line key={l} x1={0} x2={(shoreLon(l) != null ? p(shoreLon(l)!, l)[0] : W) + 2} y1={y} y2={y} className={s.zoneLine} />;
        })}
        {zone ? (
          <text x={12} y={zoneLabelY} className={s.zoneText}>
            Zone {zone.zone}
          </text>
        ) : null}
        {buoy
          ? (() => {
              const [x, y] = p(buoy.lon, buoy.lat);
              return (
                <g className={s.buoy}>
                  <circle cx={x} cy={y} r={4.5} />
                  <text x={x - 9} y={y + 4} textAnchor="end" className={s.label}>
                    {`${buoy.id}${buoy.waveFt != null ? ` · ${buoy.waveFt.toFixed(1)} ft` : buoy.windKt != null ? ` · ${Math.round(buoy.windKt)} kt` : ""}`}
                  </text>
                </g>
              );
            })()
          : null}
        {test
          ? (() => {
              const [x, y] = p(test.lon, test.lat);
              return (
                <g>
                  <rect x={x - 5} y={y - 5} width={10} height={10} transform={`rotate(45 ${x} ${y})`} className={test.old ? s.testOld : s.test} />
                  <text x={x + 10} y={y - 9} textAnchor={x > W * 0.6 ? "end" : "start"} dx={x > W * 0.6 ? -20 : 0} className={s.label}>
                    {`CDPH ${test.max != null ? `${test.max}${test.action != null ? ` of ${test.action}` : ""} ppm` : "test"}`}
                  </text>
                </g>
              );
            })()
          : null}
        <g>
          <circle cx={px} cy={py} r={7} className={s.portHalo} />
          <circle cx={px} cy={py} r={4} className={s.port} />
          <text x={px - 12} y={py + 4} textAnchor="end" className={s.portText}>
            {port.name}
          </text>
        </g>
      </svg>
      <figcaption className={s.caption}>
        <span>Not for navigation</span>
        {geo.takenAt ? <span>{`${geo.mode === "saved" ? "Read" : "As saved"} ${day(geo.takenAt)}`}</span> : null}
      </figcaption>
    </figure>
  );
}

function PlateEmpty({ className }: { className?: string }) {
  return (
    <figure className={`${s.plate} ${s.plateEmpty} ${className ?? ""}`} data-testid="trip-chart-plate">
      <div className={s.emptyWater} aria-hidden="true" />
      <figcaption className={s.caption}>
        <span>No port chosen for this trip</span>
      </figcaption>
    </figure>
  );
}
