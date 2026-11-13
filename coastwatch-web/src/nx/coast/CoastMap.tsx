"use client";

import { useMemo } from "react";
import { ChartMarks, type ChartMark } from "@/nx/chart/Labels";
import type { Bounds } from "@/nx/chart/mercator";
import { ChartRegulation, RAMP_ZONES, rampZoneAt, shoreLon } from "@/nx/chart/regulation";
import { ChartScene, type ChartScene as Scene } from "@/nx/chart/scene";
import { MapLayers, Micro, usePhone } from "@/nx/ds";
import { day, time } from "@/nx/ds/format";
import type { ShellPort } from "@/nx/shell/port";
import { portScene } from "@/nx/shell/views";
import type { CoastGates, CoastSea } from "./brief";
import type { CoastFishery } from "./fisheries";
import { zoneCaption } from "./pass";
import type { Loadable } from "./useCoast";
import s from "./coast.module.css";

type LngLat = [number, number];

/**
 * My Coast on the chart (v14 synthesis, trip frames): what is true about this trip, drawn where it is true. The port,
 * its whale zone tinted and named with its state in the records' words, the depth, the buoy's reading where the buoy
 * is (Sea now) and the last CDPH test where CDPH collected it (Toxin tests). No route, distance or ground is drawn.
 * All four layers start on, for the trip and its brief alike.
 *
 * The view frames the meaningful features: the port, the nearer limit of its whale zone at the shore, the buoy and the
 * test site, leaning seaward. On a phone that is the whole strip above the resting sheet (design issue 2); on a wide
 * screen the pass covers the foot of the chart and the camera fits the same features above it.
 */
export function CoastMap({ port, fishery, gates, sea, view }: { port: ShellPort | null; fishery: CoastFishery | null; gates: Loadable<CoastGates>; sea: Loadable<CoastSea>; view: "brief" | "trip" }) {
  const phone = usePhone();
  const g = gates.state === "ok" ? gates.data : null;
  const v = sea.state === "ok" && sea.data.sea.state === "ok" ? sea.data.sea : null;
  const buoy = v?.buoy?.lat != null && v.buoy.lon != null ? v.buoy : null;
  const test = g?.test?.lat != null && g.test.lon != null ? g.test : null;
  const ramp = !!fishery?.ramp && !!port;
  const zone = ramp && port ? rampZoneAt(port.lat) : null;

  const scene = useMemo((): Scene => tripScene(port, view, zone, [buoy ? [buoy.lon!, buoy.lat!] : null, test ? [test.lon!, test.lat!] : null]), [port, view, zone, buoy, test]);

  const caption = ramp ? zoneCaption(g?.season ?? null, g?.whale ?? null) : null;
  const marks: ChartMark[] = [];
  if (buoy && buoy.at && (buoy.waveFt != null || buoy.windKt != null)) {
    const waves = buoy.waveFt != null;
    const water = buoy.waterC != null ? Math.round((buoy.waterC * 9) / 5 + 32) : null;
    marks.push({
      id: `coast-buoy-${buoy.id}`,
      lon: buoy.lon!,
      lat: buoy.lat!,
      layer: "sea",
      priority: 70,
      leader: true,
      offset: phone ? 14 : 22,
      anchors: ["bottom-left", "left", "top-left", "bottom", "bottom-right", "right"],
      label: `Buoy ${buoy.id}, measured ${day(buoy.at)} ${time(buoy.at)}: ${waves ? `waves ${buoy.waveFt!.toFixed(1)} ft${buoy.periodS != null ? ` at ${Math.round(buoy.periodS)} s` : ""}` : `wind ${Math.round(buoy.windKt!)} kt`}${water != null ? `, water ${water} °F` : ""}`,
      content: (
        <div className={s.tag} data-compact={phone || undefined}>
          <Micro glyph="measured">
            Buoy {buoy.id} · {time(buoy.at)}
          </Micro>
          <p className={s.tagFig}>
            {waves ? buoy.waveFt!.toFixed(1) : Math.round(buoy.windKt!)} <small>{waves ? `ft${buoy.periodS != null ? ` at ${Math.round(buoy.periodS)} s` : ""}` : `kt${buoy.windDir ? ` ${buoy.windDir}` : ""}`}</small>
          </p>
          {water != null && !phone ? <p className={s.tagSub}>Water {water} °F</p> : null}
        </div>
      ),
    });
  }
  if (test) {
    const old = test.stale || (test.ageDays ?? 0) > 30;
    const n = test.raw.length;
    marks.push({
      id: `coast-test-${test.site}`,
      lon: test.lon!,
      lat: test.lat!,
      layer: "toxin",
      priority: 80,
      leader: true,
      insist: true,
      offset: phone ? 14 : 22,
      anchors: ["top", "top-left", "top-right", "left", "bottom"],
      point: <i className={s.testPoint} />,
      label: `CDPH test, ${test.site}, ${day(test.date)}: ${test.species}, highest ${test.max ?? "not detected"}${test.action != null ? ` of ${test.action} ppm` : " ppm"}, ${n} samples${old ? ", more than 30 days ago" : ""}`,
      content: (
        <div className={s.tag} data-compact={phone || undefined}>
          <Micro glyph={old ? "old" : "measured"}>
            CDPH test · {test.site} · {day(test.date)}
          </Micro>
          <p className={s.tagFig}>
            {phone ? "" : `${test.species.replace(/ crab$/, "")}, highest `}
            {test.max == null ? "ND" : test.max} <small>{test.action != null ? `of ${test.action} ppm` : "ppm"}</small>
          </p>
          {test.max != null && test.action != null && !phone ? (
            <span className={s.tagGauge} aria-hidden="true">
              <i style={{ width: `${Math.min(100, (test.max / test.action) * 100)}%` }} />
            </span>
          ) : null}
          {!phone ? (
            <p className={s.tagSub}>
              {n} samples{test.ageDays != null ? ` · ${test.ageDays} days ago${old ? " (old)" : ""}` : ""}
            </p>
          ) : null}
        </div>
      ),
    });
  }

  return (
    <>
      <ChartScene scene={scene} />
      <MapLayers page="coast" defaults={{ regulation: true, depth: true, sea: true, toxin: true }} />
      {/* the chosen fishery's regulation on the chart: RAMP's zone lines for crab, the port's zone tinted and named */}
      <ChartRegulation spec={ramp ? { zones: { highlight: zone, caption } } : null} />
      {/* a test the record gives no place for is not drawn: never a guessed position */}
      <ChartMarks marks={marks} />
    </>
  );
}

/**
 * The trip's view: the port, the nearer limit of its RAMP zone where it meets the shore (when within about 80 km),
 * the buoy and the test site, with sea west of them. Without a port, the whole coast.
 */
export function tripScene(port: ShellPort | null, view: string, zone: string | null, extra: Array<LngLat | null>): Scene {
  const base = portScene(`coast-${view}`, port, 30);
  if (!port) return base;
  const pts: LngLat[] = [[port.lon, port.lat]];
  for (const p of extra) if (p && Math.abs(p[1] - port.lat) < 0.8 && Math.abs(p[0] - port.lon) < 1) pts.push(p);
  const z = RAMP_ZONES.find((x) => x.zone === zone);
  if (z) {
    const lat = port.lat - z.south < z.north - port.lat ? z.south : z.north;
    if (Math.abs(lat - port.lat) < 0.75) pts.push([shoreLon(lat) ?? port.lon, lat]);
  }
  let w = Math.min(...pts.map((p) => p[0])), e = Math.max(...pts.map((p) => p[0]));
  let sth = Math.min(...pts.map((p) => p[1])), nth = Math.max(...pts.map((p) => p[1]));
  // at least about 40 km tall, and the sea before the coast: half as much again of water to the west
  const minLat = 0.36;
  if (nth - sth < minLat) {
    const c = (nth + sth) / 2;
    sth = c - minLat / 2;
    nth = c + minLat / 2;
  }
  const padLat = (nth - sth) * 0.1;
  w -= Math.max(0.18, (e - w) * 0.5);
  e += 0.06;
  const bounds: Bounds = [
    [w, sth - padLat],
    [e, nth + padLat],
  ];
  return { ...base, id: `coast-${view}:${port.key}:${zone ?? "port"}`, view: { bounds, maxZoom: 11 } };
}
