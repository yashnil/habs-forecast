"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { age, day, fixed, sig, time } from "@/nx/ds/format";
import type { BuoyObs, Part } from "@/lib/conditions";
import type { ExploreBuoy, ExploreStation } from "./model";
import { titleCase, type StationPick } from "./StationPoints";
import s from "./explore.module.css";

/**
 * What one station measured, from its own record: a CalHABMAP pier's newest sample per variable (dated, with its
 * age), or an NDBC buoy's newest reading (read live, each value with its own time). An old sample says nothing
 * about today; a buoy that has not reported says so. Never a closure, safety or fishing statement.
 */
const OLD_DAYS = 14;
const daysBetween = (a: string, b: string) => Math.floor((Date.parse(b) - Date.parse(`${a.slice(0, 10)}T12:00:00Z`)) / 86_400_000);

/** "40 min ago", "3 h ago": a buoy's reading is hours old at most (older ones are not read). */
export function hoursAgo(at: string, now: string): string {
  const m = Math.max(0, Math.round((Date.parse(now) - Date.parse(at)) / 60_000));
  return m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
}

function sampleValue(x: ExploreStation["latest"][number]): string {
  if (x.value === null) return "no value";
  if (x.value === 0) return "reported 0 (not quantified)";
  return `${sig(x.value, 2)} ${x.units ?? ""}`.trim();
}

function Close({ onClose }: { onClose: () => void }) {
  return (
    // design-lint-allow raw-button: the readout's close mark, the same 44 px target as ds Readout's
    <button type="button" className={s.stationClose} onClick={onClose} aria-label="Close this station">
      <span aria-hidden="true">×</span>
    </button>
  );
}

export function ShoreReadout({ station, now, place, onClose }: { station: ExploreStation; now: string; place: string | null; onClose: () => void }) {
  const rows = [...station.latest].sort((a, b) => b.date.localeCompare(a.date));
  const newest = rows[0]?.date ?? null;
  const q = new URLSearchParams({ station: station.id });
  if (place) q.set("place", place);
  return (
    <section className={s.stationCard} aria-label={`${station.name}, latest samples`} data-testid="nx-station-readout" data-kind="shore">
      <header className={s.stationHead}>
        <div>
          <p className={s.stationKind}>Observation · CalHABMAP shore station</p>
          <h2 className={s.stationTitle}>{station.name}</h2>
        </div>
        <Close onClose={onClose} />
      </header>
      {rows.length ? (
        <dl className={s.stationRows}>
          {rows.map((x) => {
            const old = daysBetween(x.date, now) > OLD_DAYS;
            return (
              <div key={x.variable} data-variable={x.variable}>
                <dt>{x.label ?? x.variable}</dt>
                <dd>
                  <b data-testid="nx-station-value">{sampleValue(x)}</b>
                  <span className={old ? s.stationOld : undefined}>
                    {day(x.date)}, {age(x.date, now)}
                  </span>
                </dd>
              </div>
            );
          })}
        </dl>
      ) : (
        <p className={s.stationNote}>No toxin or Pseudo-nitzschia sample in this station&rsquo;s published record.</p>
      )}
      <p className={s.stationNote}>
        Water samples at the pier, not shellfish or fish.{newest && daysBetween(newest, now) > OLD_DAYS ? " The newest is more than two weeks old: it says nothing about today." : " One sample describes its own day."}
      </p>
      <Link className={s.recordLink} href={`/research?${q.toString()}`} data-testid="nx-station-record">
        The whole record in Research
      </Link>
    </section>
  );
}

type BuoyPart = Part<BuoyObs> | { status: "loading" } | { status: "failed" };

export function BuoyReadout({ buoy, now, onClose }: { buoy: ExploreBuoy; now: string; onClose: () => void }) {
  const [part, setPart] = useState<BuoyPart>({ status: "loading" });
  useEffect(() => {
    let live = true;
    setPart({ status: "loading" });
    fetch(`/explore/buoy?id=${encodeURIComponent(buoy.id)}`)
      .then((r) => r.json() as Promise<Part<BuoyObs>>)
      .then((p) => live && setPart(p))
      .catch(() => live && setPart({ status: "failed" }));
    return () => {
      live = false;
    };
  }, [buoy.id]);
  const d = part.status === "ok" ? part.data : null;
  const row = (label: string, o: { value: number; at: string } | undefined, words: (v: number) => string) =>
    o ? (
      <div key={label}>
        <dt>{label}</dt>
        <dd>
          <b data-testid="nx-station-value">{words(o.value)}</b>
          <span>
            {time(o.at)}, {hoursAgo(o.at, now)}
          </span>
        </dd>
      </div>
    ) : null;
  return (
    <section className={s.stationCard} aria-label={`NDBC station ${buoy.id}, latest reading`} aria-busy={part.status === "loading"} data-testid="nx-station-readout" data-kind="buoy">
      <header className={s.stationHead}>
        <div>
          <p className={s.stationKind}>Observation · NOAA NDBC {buoy.type === "fixed" ? "station" : "buoy"} {buoy.id}</p>
          <h2 className={s.stationTitle}>{titleCase(buoy.name)}</h2>
        </div>
        <Close onClose={onClose} />
      </header>
      {part.status === "loading" ? (
        <p className={s.stationNote}>Reading NDBC…</p>
      ) : part.status === "failed" ? (
        <p className={s.stationNote}>This station couldn&rsquo;t be read. Check the connection and open it again.</p>
      ) : part.status !== "ok" ? (
        <p className={s.stationNote} data-testid="nx-station-none">
          {part.reason} Nothing is shown in its place.
        </p>
      ) : (
        <dl className={s.stationRows}>
          {row("Wind", d!.windKt, (v) => `${sig(v, 2)} kt${d!.windDirDeg ? ` from ${Math.round(d!.windDirDeg.value)}°` : ""}${d!.gustKt ? `, gusts ${sig(d!.gustKt.value, 2)}` : ""}`)}
          {row("Waves", d!.waveFt, (v) => `${fixed(v, 1)} ft${d!.periodS ? ` at ${Math.round(d!.periodS.value)} s` : ""}`)}
          {row("Water temperature", d!.waterC, (v) => `${fixed(v * 1.8 + 32, 1)} °F (${fixed(v, 1)} °C)`)}
          {row("Air temperature", d!.airC, (v) => `${fixed(v * 1.8 + 32, 1)} °F`)}
        </dl>
      )}
      <p className={s.stationNote}>Read from NDBC just now; each value carries its own time{part.status === "ok" ? ` (${day(part.fetchedAt)})` : ""}.</p>
      {part.status === "ok" ? (
        <a className={s.recordLink} href={part.url} target="_blank" rel="noreferrer">
          NDBC station page ↗
        </a>
      ) : null}
    </section>
  );
}

export function StationReadout({ pick, stations, buoys, now, place, onClose }: { pick: StationPick; stations: ExploreStation[]; buoys: ExploreBuoy[]; now: string; place: string | null; onClose: () => void }) {
  if (pick.kind === "shore") {
    const st = stations.find((x) => x.id === pick.id);
    return st ? <ShoreReadout station={st} now={now} place={place} onClose={onClose} /> : null;
  }
  const b = buoys.find((x) => x.id === pick.id);
  return b ? <BuoyReadout buoy={b} now={now} onClose={onClose} /> : null;
}
