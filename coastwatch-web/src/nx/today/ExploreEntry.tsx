"use client";

import Link from "next/link";
import COAST from "@/nx/explore/locator-coast.json";
import { exploreHref } from "@/nx/explore/links";
import s from "./today.module.css";

/** The coast's outline, drawn small (the same line as Explore's locator). */
const LON0 = -125.2, LON1 = -116.6, LAT0 = 32.2, LAT1 = 42.2, W = 30;
const H = Math.round((W * (LAT1 - LAT0)) / ((LON1 - LON0) * Math.cos((37 * Math.PI) / 180)));
const D = (COAST as [number, number][]).map(([lon, lat], i) => `${i ? "L" : "M"}${(((lon - LON0) / (LON1 - LON0)) * W).toFixed(1)},${(((LAT1 - lat) / (LAT1 - LAT0)) * H).toFixed(1)}`).join("");

/**
 * Today's way into Explore California Waters (v15.1): one clear entry in the ticket, never a chip or a menu item.
 * It carries the port, so the statewide view keeps it marked and "Back to Today" returns here.
 */
export function ExploreEntry({ place, lon, lat }: { place: string; lon: number; lat: number }) {
  const x = ((lon - LON0) / (LON1 - LON0)) * W, y = ((LAT1 - lat) / (LAT1 - LAT0)) * H;
  return (
    <Link href={exploreHref({ view: "coast", place })} className={s.exploreEntry} data-testid="nx-today-explore">
      <svg viewBox={`-3 -3 ${W + 6} ${H + 6}`} width={W + 6} height={H + 6} aria-hidden="true" className={s.exploreMap}>
        <path d={D} />
        <circle cx={x} cy={y} r={2.6} />
      </svg>
      <span className={s.exploreWords}>
        <b>Explore California Waters</b>
        <small>Chlorophyll, C-HARM, currents and stations, Oregon to Mexico</small>
      </span>
      <span className={s.exploreGo} aria-hidden="true">
        →
      </span>
    </Link>
  );
}
