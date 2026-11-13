"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { Icon, MarginRow, MarginRows, Skeleton, Stamp, Unavailable, type Provenance } from "@/nx/ds";
import { hatchSegments } from "@/nx/ds/Mark";
import { sig } from "@/nx/ds/format";
import { cx } from "@/nx/ds/util";
import { BAR_RULE_URL, REGULATED_BARS, type CoastSea, type SeaPeriod } from "./brief";
import { risingSpans, seaDays, seasScale, seasWords, tideClock, tideFeet, tidesOn, tideWords, type SeaDay } from "./sea";
import type { Tide } from "@/lib/conditions";
import type { Loadable } from "./useCoast";
import s from "./coast.module.css";

const HATCH = hatchSegments(0, 0, 200, 20, 4);

/**
 * The sea window: what the buoys measure now, then NWS's periods by day for the reader to choose from, and
 * for the chosen day the wind as NWS stated it (its exact sentence a tap away) and the predicted tide.
 * Nothing here judges a day (T36); a regulated bar gets the rule, its radio channels and the rising tide. The buoy
 * itself is drawn on the chart by ./CoastMap (the Sea now layer), with its reading where it was measured.
 */
export function SeaWindow({ sea, placeKey }: { sea: Loadable<CoastSea>; placeKey: string }) {
  const [chosen, setChosen] = useState<string | null>(null);
  if (sea.state === "idle") return null;
  // a placeholder at the sea window's own height (the buoys' line, then the days), so the page under it holds still
  if (sea.state === "loading") return <Skeleton label="Reading NWS, the buoys and the tide…" lines={8} className={s.seaLoading} />;
  if (sea.state === "error" || sea.data.sea.state !== "ok")
    return (
      <Unavailable
        what="The sea could not be read"
        why="NWS, NDBC or CO-OPS did not answer Neritic. Their own pages have the forecast and the buoys."
        next={
          <a href="https://www.weather.gov/marine/" className={s.link}>
            NWS marine forecasts
          </a>
        }
      />
    );
  const { sea: v, now } = sea.data;
  const days = seaDays(v.lanes, now);
  const pick = days.find((d) => d.date === chosen) ?? days[0] ?? null;
  const bar = REGULATED_BARS[placeKey] ?? null;
  const nws: Provenance | null = v.issued
    ? { kind: "prediction", title: `NWS coastal waters forecast, ${v.zone}`, source: `NWS ${v.zone}`, sourceLong: `National Weather Service coastal waters forecast, zone ${v.zone}`, sourceUrl: v.marineUrl ?? undefined, date: v.issued, dateRole: "issued" }
    : null;
  const tide: Provenance | null = v.tideStation && pick
    ? { kind: "prediction", title: `Tide, ${v.tideStation.name}`, source: `NOAA CO-OPS ${v.tideStation.id}`, sourceLong: `NOAA CO-OPS tide predictions, station ${v.tideStation.id} ${v.tideStation.name}, feet above MLLW`, sourceUrl: v.tideStation.url, date: pick.date, dateRole: "predicted", caveats: ["Astronomical prediction: wind and swell move the real water."] }
    : null;
  return (
    <>
      <MarginRows label="Sea window" className={s.rows}>
        {v.alerts.map((a) => (
          <MarginRow
            key={a.event}
            label="Advisory"
            value={
              <a href={a.url} className={s.link}>
                {a.headline || a.event}
              </a>
            }
          />
        ))}
        {v.buoy ? (
          <MarginRow
            label="Now"
            value={<Measured v={v} />}
            stamp={
              <span className={s.stamps}>
                <Stamp now={now} provenance={{ kind: "measured", title: `Waves, buoy ${v.buoy.id}`, source: `NDBC buoy ${v.buoy.id}, ${Math.round(v.buoy.km)} km`, sourceLong: `NDBC ${v.buoy.id} ${v.buoy.name}`, sourceUrl: v.buoy.url, date: v.buoy.at ?? now }} />
                {v.windBuoy ? <Stamp now={now} provenance={{ kind: "measured", title: `Wind, buoy ${v.windBuoy.id}`, source: `NDBC ${v.windBuoy.id}, ${Math.round(v.windBuoy.km)} km`, sourceLong: `NDBC ${v.windBuoy.id} ${v.windBuoy.name}: the nearest station that measures wind`, sourceUrl: v.windBuoy.url, date: v.windBuoy.at }} /> : null}
              </span>
            }
          />
        ) : (
          <MarginRow label="Now" state="unavailable" note={v.buoyReason ?? "No buoy reading near this port."} />
        )}
      </MarginRows>
      {days.length ? (
        <>
          <DayStrip days={days} chosen={pick?.date ?? null} onChoose={setChosen} />
          {nws ? (
            <p className={s.stripStamp}>
              <Stamp now={now} provenance={nws} />
            </p>
          ) : null}
        </>
      ) : (
        <p className={s.quiet}>{v.marineReason ?? "NWS has no forecast periods for these waters right now."}</p>
      )}
      {pick ? (
        <MarginRows label={`${pick.label}, forecast and tide`} className={s.rowsTight}>
          <Period label="Day" p={pick.day} />
          <Period label="Night" p={pick.night} />
          <MarginRow label="Tide" value={tideWords(v.tides, pick.date) ? <TideTable tides={tidesOn(v.tides, pick.date)} /> : <span className={s.quiet}>{v.tideReason ?? "No predictions for this day."}</span>} stamp={tide ? <Stamp now={now} provenance={tide} /> : undefined} />
          {bar ? (
            <MarginRow
              label="Bar"
              value={
                <>
                  <span>
                    {bar.name}: a regulated crossing. {risingSpans(v.tides, pick.date).length ? `Tide rising ${risingSpans(v.tides, pick.date).join(", ")}. ` : ""}Restrictions and closures are broadcast on VHF 16 and 22A.
                  </span>
                  <span className={s.links}>
                    <a href={BAR_RULE_URL} className={s.link}>
                      The rule, 33 CFR 165.1196
                    </a>
                    {bar.guide ? (
                      <a href={bar.guide} className={s.link}>
                        NWS bar guide
                      </a>
                    ) : null}
                  </span>
                </>
              }
            />
          ) : null}
        </MarginRows>
      ) : null}
    </>
  );
}

/** The day's highs and lows, one per line: kind, height, time. */
function TideTable({ tides }: { tides: Tide[] }) {
  return (
    <span className={s.tides} role="list">
      {tides.map((t) => (
        <span key={t.at} role="listitem" className={s.tide}>
          <span>{t.type === "H" ? "High" : "Low"}</span>
          <span className={s.tideFt}>{tideFeet(t.ft)}</span>
          <span>{tideClock(t.at)}</span>
        </span>
      ))}
    </span>
  );
}

function Measured({ v }: { v: CoastSea["sea"] }) {
  const b = v.buoy!;
  const parts = [b.waveFt != null ? `waves ${sig(b.waveFt)}\u00a0ft${b.periodS != null ? ` at ${Math.round(b.periodS)}\u00a0s` : ""}` : null, b.windKt != null ? `wind ${b.windDir ?? ""} ${Math.round(b.windKt)}\u00a0kt`.replace("  ", " ") : v.windBuoy ? `wind ${v.windBuoy.windDir ?? ""} ${Math.round(v.windBuoy.windKt)}\u00a0kt`.replace("  ", " ") : null].filter(Boolean);
  const t = parts.join(", ");
  return <span>{t ? t.charAt(0).toUpperCase() + t.slice(1) : "No wave or wind value in the last hours."}</span>;
}

/** One NWS period: the wind as stated; NWS's own sentence opens in place. */
function Period({ label, p }: { label: string; p: SeaPeriod | null }) {
  if (!p) return null;
  return (
    <MarginRow
      label={label}
      value={p.wind ?? "No wind stated"}
      detail={
        <blockquote className={s.quote}>
          <span className={s.quoteName}>{p.name}.</span> {p.text}
        </blockquote>
      }
    />
  );
}

/**
 * The forecast's days, one row each, chosen like a radio group (arrows move and choose). Seas are drawn as
 * hatched bars (a forecast mark, never solid) with the range NWS stated.
 */
function DayStrip({ days, chosen, onChoose }: { days: SeaDay[]; chosen: string | null; onChoose: (d: string) => void }) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const max = seasScale(days);
  const at = Math.max(0, days.findIndex((d) => d.date === chosen));
  const key = (i: number) => (e: KeyboardEvent<HTMLButtonElement>) => {
    const d = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const n = (i + d + days.length) % days.length;
    onChoose(days[n].date);
    refs.current[n]?.focus();
  };
  const cell = (p: SeaPeriod | null) => {
    const w = seasWords(p);
    if (!p || w === null) return <span className={s.dayCell} />;
    return (
      <span className={s.dayCell}>
        <span className={s.seasTrack}>
          <span className={s.seasBar} style={{ width: `${(p.seasHi! / max) * 100}%` }}>
            <svg focusable="false" aria-hidden>
              <path d={HATCH} />
            </svg>
          </span>
        </span>
        <span className={s.seasNum}>{w}</span>
      </span>
    );
  };
  return (
    <div className={s.strip}>
      <div className={s.stripHead} aria-hidden>
        <span>Seas, ft</span>
        <span>Day</span>
        <span>Night</span>
      </div>
      <div role="radiogroup" aria-label="Day to read" className={s.stripRows}>
        {days.map((d, i) => {
          const on = i === at;
          const words = [d.day && seasWords(d.day) ? `day seas ${seasWords(d.day)} ft` : null, d.night && seasWords(d.night) ? `night seas ${seasWords(d.night)} ft` : null].filter(Boolean).join(", ");
          return (
            <button
              key={d.date}
              ref={(el) => {
                refs.current[i] = el;
              }}
              type="button"
              role="radio"
              aria-checked={on}
              aria-label={`${d.label}${words ? `: ${words}` : ""}`}
              tabIndex={on ? 0 : -1}
              className={cx(s.dayRow, on && s.dayOn)}
              onClick={() => onChoose(d.date)}
              onKeyDown={key(i)}
              data-testid={`sea-day-${d.date}`}
            >
              <span className={s.dayName}>{d.label}</span>
              {cell(d.day)}
              {cell(d.night)}
            </button>
          );
        })}
      </div>
    </div>
  );
}
