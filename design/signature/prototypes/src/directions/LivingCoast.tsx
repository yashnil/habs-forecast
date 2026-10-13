import { fmtKm } from "../lib/geo";
import { fmtDay, relDay } from "../lib/time";
import { DirMap } from "./DirMap";
import { facts, StatusLine } from "./facts";
import { type DirState, fmtLat, POINT } from "./shared";

/**
 * Direction B — Living Coast. Calm and light: the coast in soft blue-greys and sand; the forecast
 * only around the chosen point (cells outside the lens are not drawn); content as plain sentences
 * in one floating sheet; one selection ripple on arrival, then stillness. No water animation.
 */
const COLOURS = { official: "#c27a12", station: "#0c6f62", stationFill: "#2a9d86", port: "#20303f", portLabel: "#20303f", halo: "rgba(246,243,236,0.95)", edge: "#7a52c0", select: "#20303f" };

export function LivingCoast({ st, mobile }: { st: DirState; mobile: boolean }) {
  const f = facts(st);
  const callouts = [{ id: "sel", at: POINT, node: <span className="lc-ripple" aria-hidden="true"><i /><b /></span> }];
  return (
    <div className="lc" data-mobile={mobile || undefined}>
      <DirMap v="b" st={st} field={{ mode: "lens", lens: { lon: POINT[0], lat: POINT[1], km: 14 }, outside: 0 }} center={[-122.0, 36.8]} zoom={mobile ? 8.6 : 9.3}
        threads={false} hatch={false} callouts={callouts} colours={COLOURS}
        padding={mobile ? { top: 90, bottom: 380, left: 10, right: 10 } : { top: 80, bottom: 40, left: 520, right: 40 }} />
      <header className="lc-top">
        <span className="lc-brand">CoastWatch</span>
        <StatusLine className="lc-status" />
      </header>
      <article className="lc-sheet">
        <p className="lc-kicker">Your waters · {fmtDay(f.validFor, { weekday: true })}</p>
        <h1 className="lc-place">{f.place}</h1>
        <div className="lc-lines">
          <p><span className="mark mark-rule" aria-hidden="true" /><span><b>{f.nCoverWord} official notices</b> cover these waters: {f.coverSummary}.{f.limit ? ` The ${f.limitSpecies} ${f.limitSharedBy > 1 ? "notices end" : "notice ends"} at ${fmtLat(f.limit.lat)}, ${fmtKm(f.limit.km)} ${f.limit.which === "south" ? "south" : "north"} of here.` : ""}</span></p>
          <p><span className="mark mark-sample" aria-hidden="true" /><span><b>{f.station.station.name}</b>, {fmtKm(f.station.km)} {f.station.bearing}, was last sampled on {fmtDay(f.station.lastDate!)} — {f.station.ageDays} days ago.</span></p>
          {f.cls && <p><span className="mark mark-field" aria-hidden="true" /><span>NOAA's model gives a <b>{f.cls.lo}–{f.cls.hi} % chance</b> of high toxin in the water for {relDay(f.validFor)}, from a run made {fmtDay(f.issued)} that CoastWatch could not confirm is the newest.</span></p>}
          <p><span className="mark mark-gap" aria-hidden="true" /><span>Whether seafood here is safe isn't something this map can tell you. Only CDPH testing does.</span></p>
        </div>
        {f.disputed && <p className="lc-disp">{f.disputed.by.agency} {f.disputed.by.release} appears to lift “{f.disputed.record.title}”. Until someone reviews it, it isn't shown as in effect. <a href={f.disputed.by.url} target="_blank" rel="noopener" className="ext">Read it</a></p>}
        <p className="lc-lens">The violet patch is NOAA's forecast within 14 km of your point, cell by cell. Elsewhere it is hidden, not absent.</p>
      </article>
    </div>
  );
}
