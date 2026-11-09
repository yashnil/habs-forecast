"use client";

import type { SeaView } from "@/pilot/facts";
import { alertFactId, dayWords, nextTide, pacificDay } from "@/pilot/facts";
import { periodLine } from "./marine";
import { TideCurve } from "./Charts";
import { FactAnchors, Section, factProps } from "./ui";

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

/** Sea and weather: the daily-habit layer. NWS's forecast as lanes, the nearest buoy now, the tide, the water. */
export function SeaWeather({ sea, now, lede, place, factIds = [] }: { sea: SeaView; now: string; lede: string | null; place: string; /** the ids of the sea, weather and tide facts drawn here (their addresses) */ factIds?: string[] }) {
  const ids = (prefix: string) => factIds.filter((id) => id.startsWith(prefix));
  const day = pacificDay(now);
  const [hh, mm] = PT_HM.format(new Date(now)).split(":").map(Number);
  const nowMin = hh * 60 + mm;
  const maxWind = Math.max(25, ...sea.lanes.map((p) => p.windHi ?? 0));
  const nt = nextTide(sea.tides, now);
  const todays = sea.tides.filter((t) => t.at.startsWith(day));
  return (
    <Section id="sea" area="sea" title="Sea & weather" aside={sea.zone ? <span className="today-src">NWS {sea.zone}</span> : undefined} feedback={sea.state === "ok" && factIds.length ? { factId: factIds.includes("weather:nws") ? "weather:nws" : factIds[0], placeId: place } : null}>
      {sea.state === "loading" && (
        <div aria-busy="true" aria-live="polite" data-testid="sea-loading">
          <p className="sr-only">Loading the marine forecast, buoy and tide…</p>
          <div className="today-skel mt-4 h-5 w-3/4" />
          <div className="today-skel mt-4 h-[120px] w-full" />
          <div className="today-skel mt-3 h-[60px] w-full" />
        </div>
      )}
      {sea.state === "error" && (
        <p className="today-say" role="status" data-testid="sea-error">
          Sea and weather could not be loaded. Read the{" "}
          <a className="nx-focus underline underline-offset-4" href="https://www.weather.gov/marine/" target="_blank" rel="noreferrer">
            NWS marine forecast
          </a>{" "}
          and{" "}
          <a className="nx-focus underline underline-offset-4" href="https://www.ndbc.noaa.gov/" target="_blank" rel="noreferrer">
            NDBC buoys
          </a>{" "}
          directly.
        </p>
      )}
      {sea.state === "ok" && (
        <>
          {sea.alerts.map((a) => (
            <p key={a.event} className="mt-3 flex items-start gap-2 border-l-2 border-[var(--t-amber)] pl-3 text-[15px] font-medium text-n-ink" data-testid="sea-alert" {...factProps(alertFactId(a.event))}>
              <span className="t-amber">NWS:</span> {a.headline}
            </p>
          ))}
          <div {...factProps("weather:nws")}>
          <FactAnchors ids={ids("weather:nws:")} />
          {lede ? <p className="today-say">{lede}</p> : sea.marineReason && <p className="today-say">{sea.marineReason}</p>}
          {sea.lanes.length > 0 && (
            <div className="mt-3 overflow-hidden">
              <table className="today-lanes" data-testid="sea-lanes">
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
                The number is the highest wind NWS states for the period; a bar is solid to the low end of its range and lighter to the top. Forecasters&rsquo; words in full:{" "}
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
          </div>

          <div data-fact-card="">
          <FactAnchors ids={factIds.filter((id) => id.startsWith("sea:"))} />
          <div className="today-readouts" data-testid="sea-now">
            <div>
              <p className="nx-caps">Wind now</p>
              {sea.buoy?.windKt != null ? (
                <p className="today-v mt-1.5">
                  {Math.round(sea.buoy.windKt)}
                  <small>kt {sea.buoy.windDir ?? ""}</small>
                </p>
              ) : (
                <p className="today-v mt-1.5 text-n-ink-3">—</p>
              )}
              <p className="today-src mt-1.5">{sea.buoy?.windKt != null ? (sea.buoy.gustKt != null ? `gust ${Math.round(sea.buoy.gustKt)}` : "measured") : "not measured here"}</p>
            </div>
            <div>
              <p className="nx-caps">Waves now</p>
              {sea.buoy?.waveFt != null ? (
                <p className="today-v mt-1.5">
                  {sea.buoy.waveFt.toFixed(1)}
                  <small>ft</small>
                </p>
              ) : (
                <p className="today-v mt-1.5 text-n-ink-3">—</p>
              )}
              <p className="today-src mt-1.5">{sea.buoy?.periodS != null ? `${Math.round(sea.buoy.periodS)} s period` : "no wave reading"}</p>
            </div>
            <div>
              <p className="nx-caps">Water</p>
              {sea.water ? (
                <p className="today-v mt-1.5">
                  {Math.round((sea.water.c * 9) / 5 + 32)}
                  <small>°F</small>
                </p>
              ) : (
                <p className="today-v mt-1.5 text-n-ink-3">—</p>
              )}
              <p className="today-src mt-1.5">{sea.water ? `${sea.water.c.toFixed(1)} °C ${sea.water.where}` : "no reading nearby"}</p>
            </div>
          </div>
          <p className="today-src mt-2">
            {sea.buoy ? (
              <>
                Measured at{" "}
                <a href={sea.buoy.url} target="_blank" rel="noreferrer" className="nx-focus">
                  NDBC buoy {sea.buoy.id}
                </a>
                , {Math.round(sea.buoy.km)} km away{sea.buoy.at ? `, ${ago(sea.buoy.at, now)}` : ""}. A reading at one point, not a forecast.
              </>
            ) : (
              (sea.buoyReason ?? "No buoy reading.")
            )}
          </p>
          </div>

          <div className="mt-5" data-testid="sea-tide" data-fact-card="">
            <FactAnchors ids={ids("tide:")} />
            <div className="flex items-baseline justify-between gap-3">
              <p className="nx-caps">Tide today{sea.tideStation ? ` · ${sea.tideStation.name}` : ""}</p>
              {nt && (
                <p className="text-[14px] text-n-ink">
                  Next: <b className="font-semibold">{nt.type === "H" ? "high" : "low"}</b> {nt.ft.toFixed(1)} ft at {tclock(nt.at)}
                </p>
              )}
            </div>
            {sea.tides.length ? (
              <>
                <TideCurve tides={sea.tides} day={day} nowMin={nowMin} />
                <ul className="flex flex-wrap gap-x-4 gap-y-1 font-data text-[13px] text-n-ink-2">
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
                  , feet above MLLW. Astronomical predictions; wind and swell shift real water levels.
                </p>
              </>
            ) : (
              <p className="today-src mt-1">{sea.tideReason ?? "No tide predictions."}</p>
            )}
          </div>
        </>
      )}
    </Section>
  );
}
