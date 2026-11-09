"use client";

import Link from "next/link";
import type { SeaView } from "@/pilot/facts";
import { alertEndWords, alertFactId, dayWords, nextTide, pacificDay } from "@/pilot/facts";
import { Fold } from "@/v4/motion";
import { Figure } from "@/v4/ui/Figure";
import { Tag } from "@/v4/ui/Tag";
import { periodLine } from "./marine";
import { SeaStrip } from "./Charts";
import { Chapter, FactAnchors, factProps } from "./ui";

const PT_CLOCK = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit" });
const PT_HM = new Intl.DateTimeFormat("en-GB", { timeZone: "America/Los_Angeles", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const clock = (iso: string) => PT_CLOCK.format(new Date(iso));
const tclock = (at: string) => {
  const m = /(\d{2}):(\d{2})/.exec(at);
  if (!m) return at;
  const h = Number(m[1]);
  return `${h % 12 || 12}:${m[2]} ${h < 12 ? "AM" : "PM"}`;
};
const ago = (iso: string, now: string) => {
  const m = Math.max(0, Math.round((Date.parse(now) - Date.parse(iso)) / 60000));
  return m < 60 ? `${m} min ago` : m < 48 * 60 ? `${Math.floor(m / 60)} h ago` : dayWords(iso, now);
};

/** Sea, chapter two: an advisory band, three figures, then the next 36 hours; the forecasters' full periods, the buoy and the tide table fold away. */
export function SeaWeather({ sea, now, seaLine, place, factIds = [] }: { sea: SeaView; now: string; seaLine: string | null; place: string; /** the ids of the sea, weather and tide facts drawn here (their addresses) */ factIds?: string[] }) {
  const ids = (prefix: string) => factIds.filter((id) => id.startsWith(prefix));
  const day = pacificDay(now);
  const [hh, mm] = PT_HM.format(new Date(now)).split(":").map(Number);
  const nowMin = hh * 60 + mm;
  const maxWind = Math.max(25, ...sea.lanes.map((p) => p.windHi ?? 0));
  const nt = nextTide(sea.tides, now);
  const todays = sea.tides.filter((t) => t.at.startsWith(day));
  // wind at the nearest buoy, else at the nearest buoy that measures it (named, with its distance)
  const wind = sea.buoy?.windKt != null ? { windKt: sea.buoy.windKt, windDir: sea.buoy.windDir, gustKt: sea.buoy.gustKt } : sea.windBuoy;
  return (
    <Chapter id="sea" area="sea" title="Sea" testid="today-sea" say={sea.state === "ok" ? (seaLine ?? sea.marineReason) : undefined}>
      {sea.state === "loading" && (
        <div aria-busy="true" aria-live="polite" data-testid="sea-loading">
          <p className="sr-only">Loading the marine forecast, buoy and tide…</p>
          <div className="today-skel h-[92px] w-full" />
          <div className="today-skel mt-4 h-[214px] w-full" />
        </div>
      )}
      {sea.state === "error" && (
        <p className="today-say" role="status" data-testid="sea-error">
          Sea and weather could not be loaded. Read the{" "}
          <a className="nx-link nx-focus" href="https://www.weather.gov/marine/" target="_blank" rel="noreferrer">
            NWS marine forecast
          </a>{" "}
          and{" "}
          <a className="nx-link nx-focus" href="https://www.ndbc.noaa.gov/" target="_blank" rel="noreferrer">
            NDBC buoys
          </a>{" "}
          directly.
        </p>
      )}
      {sea.state === "ok" && (
        <>
          {sea.alerts.map((a) => (
            <p key={a.event} className="today-band" data-testid="sea-alert" {...factProps(alertFactId(a.event))}>
              <Tag level="watch">NWS</Tag>
              <span>
                {a.event}
                <span className="text-n-ink-2"> · {alertEndWords(a.ends, now)}</span>
              </span>
            </p>
          ))}
          <div data-fact-card="">
            <FactAnchors ids={factIds.filter((id) => id.startsWith("sea:"))} />
            <div className="today-figures" data-testid="sea-now">
              {wind ? (
                <Figure
                  label="Wind, measured"
                  value={Math.round(wind.windKt)}
                  unit={`kt ${wind.windDir ?? ""}`}
                  testId="wind-now"
                  age={
                    <span data-testid="wind-where">
                      {wind.gustKt != null ? `gust ${Math.round(wind.gustKt)} · ` : ""}
                      {sea.windBuoy ? `buoy ${sea.windBuoy.id}, ${Math.round(sea.windBuoy.km)} km away` : sea.buoy ? `buoy ${sea.buoy.id}` : "measured"}
                    </span>
                  }
                />
              ) : (
                <Figure label="Wind, measured" value="—" age={<span data-testid="wind-where">no wind reading within 60 km</span>} />
              )}
              <Figure label="Waves, measured" value={sea.buoy?.waveFt != null ? sea.buoy.waveFt.toFixed(1) : "—"} unit={sea.buoy?.waveFt != null ? "ft" : undefined} age={sea.buoy?.periodS != null ? `${Math.round(sea.buoy.periodS)} s period` : "no wave reading"} />
              <Figure label="Water" value={sea.water ? Math.round((sea.water.c * 9) / 5 + 32) : "—"} unit={sea.water ? "°F" : undefined} stale={!!sea.water?.stale} age={sea.water ? `${sea.water.c.toFixed(1)} °C, ${sea.water.where.replace(/^at /, "")}` : "no reading nearby"} />
            </div>
          </div>
          {sea.lanes.length > 0 && (
            <div className="today-strip" {...factProps("weather:nws")}>
              <SeaStrip lanes={sea.lanes} tides={sea.tides} day={day} nowMin={nowMin} />
              {nt && (
                <p className="today-note mt-2">
                  Next tide: {nt.type === "H" ? "high" : "low"} {nt.ft.toFixed(1)} ft at {tclock(nt.at)}
                </p>
              )}
            </div>
          )}
          <Link href={`/coast?place=${encodeURIComponent(place)}`} className="nx-link nx-focus today-go" data-testid="sea-limits">
            5 days against your limits
          </Link>
          <Fold testid="sea-more" title="All forecast periods, the buoy, the tide table" className="today-fold" heading="h3">
            <div data-fact-card="">
              <FactAnchors ids={ids("weather:nws:")} />
              {sea.alerts.map((a) => (
                <p key={a.url} className="mt-2 text-[15px] text-n-ink" data-testid="sea-alert-words">
                  NWS&rsquo;s headline: {a.headline}
                </p>
              ))}
              {sea.lanes.length > 0 && (
                <div className="overflow-hidden pt-2">
                  <table className="today-lanes" data-testid="sea-periods">
                    <caption className="sr-only">
                      NWS zone forecast {sea.zone}, issued {sea.issued ? clock(sea.issued) : "time unknown"}: wind in knots and seas in feet for each period, as stated by forecasters.
                    </caption>
                    <thead>
                      <tr>
                        <td />
                        {sea.lanes.map((p, i) => (
                          <th key={i} scope="col" className={p.night ? "night" : undefined} title={p.name}>
                            {p.short}
                            <small>{p.night ? "night" : "day"}</small>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="today-lane-wind">
                        <th scope="row">
                          Wind
                          <br />
                          kt
                        </th>
                        {sea.lanes.map((p, i) => (
                          <td key={i} className={p.night ? "night" : undefined} title={p.text}>
                            {p.windHi != null ? (
                              <>
                                <span aria-hidden className="today-bar" style={{ height: `${(p.windHi / maxWind) * 46}px` }}>
                                  <i style={{ height: `${((p.windLo ?? p.windHi) / Math.max(p.windHi, 1)) * 100}%` }} />
                                </span>
                                <span aria-hidden className="today-dir">{p.dir && p.dir !== "light" && p.dir !== "variable" ? p.dir : ""}</span>
                                <span className="today-lane-num">
                                  <span className="sr-only">{periodLine(p)}</span>
                                  <span aria-hidden>{p.windHi}</span>
                                </span>
                              </>
                            ) : (
                              <span className="today-lane-num">{p.dir === "light" ? "lt" : "—"}</span>
                            )}
                          </td>
                        ))}
                      </tr>
                      <tr className="today-lane-seas">
                        <th scope="row">
                          Seas
                          <br />
                          ft
                        </th>
                        {sea.lanes.map((p, i) => (
                          <td key={i} className={p.night ? "night" : undefined}>
                            {p.seasHi != null ? (p.seasLo != null && p.seasLo !== p.seasHi ? `${p.seasLo}–${p.seasHi}` : p.seasHi) : "—"}
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                  <p className="today-src mt-2">
                    Each bar is the wind range NWS states, solid to its low end; the number is the top. Forecasters&rsquo; words:{" "}
                    {sea.marineUrl ? (
                      <a href={sea.marineUrl} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center">
                        NWS {sea.zone}, issued {sea.issued ? clock(sea.issued) : ""}
                      </a>
                    ) : (
                      "NWS"
                    )}
                    .
                  </p>
                </div>
              )}
              <p className="today-note mt-3">
                {sea.buoy ? (
                  <>
                    Measured at{" "}
                    <a href={sea.buoy.url} target="_blank" rel="noreferrer" className="nx-link nx-focus">
                      NDBC buoy {sea.buoy.id}
                    </a>
                    , {Math.round(sea.buoy.km)} km away{sea.buoy.at ? `, ${ago(sea.buoy.at, now)}` : ""}
                    {sea.windBuoy && (
                      <>
                        ; it has no wind sensor, so wind is from{" "}
                        <a href={sea.windBuoy.url} target="_blank" rel="noreferrer" className="nx-link nx-focus">
                          buoy {sea.windBuoy.id}
                        </a>
                        , {Math.round(sea.windBuoy.km)} km away, {ago(sea.windBuoy.at, now)}
                      </>
                    )}
                    . {sea.windBuoy ? "Readings at single points" : "A reading at one point"}, not a forecast.
                  </>
                ) : (
                  (sea.buoyReason ?? "No buoy reading.")
                )}
              </p>
              <div className="mt-3" data-testid="sea-tide" data-fact-card="">
                <FactAnchors ids={ids("tide:")} />
                <p className="nx-caps">Tide today{sea.tideStation ? ` · ${sea.tideStation.name}` : ""}</p>
                {sea.tides.length ? (
                  <>
                    <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1 font-data text-[13px] text-n-ink-2">
                      {todays.map((t) => (
                        <li key={t.at}>
                          {t.type === "H" ? "High" : "Low"} {tclock(t.at)} · {t.ft.toFixed(1)} ft
                        </li>
                      ))}
                    </ul>
                    <p className="today-src mt-1.5">
                      NOAA CO-OPS predictions
                      {sea.tideStation && (
                        <>
                          {" "}
                          at{" "}
                          <a href={sea.tideStation.url} target="_blank" rel="noreferrer" className="nx-focus">
                            station {sea.tideStation.id}
                          </a>
                        </>
                      )}
                      , feet above MLLW. Astronomical predictions; wind and swell shift real water levels. NWS day and night periods run about 6 to 6.
                    </p>
                  </>
                ) : (
                  <p className="today-src mt-1">{sea.tideReason ?? "No tide predictions."}</p>
                )}
              </div>
            </div>
          </Fold>
        </>
      )}
    </Chapter>
  );
}
