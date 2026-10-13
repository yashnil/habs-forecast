import { fmtKm } from "../lib/geo";
import { fmtDay } from "../lib/time";
import { DirMap } from "./DirMap";
import { facts, noticeLabel, StatusLine } from "./facts";
import { type DirState, fmtLat, POINT } from "./shared";

/**
 * Direction C — Editorial Atlas. Paper-first; the map is a plate. A headline that states the
 * answer, a dateline, and margin notes keyed by number to callouts on the plate (text ↔ map).
 */
const COLOURS = { official: "#b4660a", station: "#0c6f62", stationFill: "#0c6f62", port: "#1d2b38", portLabel: "#1d2b38", halo: "rgba(240,236,226,0.95)", edge: "#6a3fb0", select: "#1d2b38" };

export function Atlas({ st, mobile }: { st: DirState; mobile: boolean }) {
  const f = facts(st);
  const stationAt: [number, number] = [f.station.station.lon, f.station.station.lat];
  const cellAt = st.reading!.model.status === "value" ? st.reading!.model.center : POINT;
  const callouts = [
    { id: "1", at: POINT, node: <span className="at-num at-num-sel">1</span> },
    { id: "2", at: stationAt, node: <span className="at-num at-num-m">2</span> },
    { id: "3", at: [cellAt[0] + 0.05, cellAt[1] - 0.05] as [number, number], node: <span className="at-num at-num-f">3</span> },
    ...(f.limit ? [{ id: "4", at: [POINT[0] - 0.12, f.limit.lat] as [number, number], node: <span className="at-num at-num-o">4</span> }] : []),
  ];
  const plate = (
    <figure className="at-plate">
      <div className="at-frame">
        <DirMap v="c" st={st} field={{ mode: "veil", veil: 0.42 }} center={[-122.02, 36.78]} zoom={mobile ? 8.5 : innerHeight < 800 ? 8.6 : 9.0} threads={false} callouts={callouts} colours={COLOURS} interactive />
        <svg className="at-north" width="22" height="40" viewBox="0 0 22 40" aria-label="North"><path d="M11 2 L17 22 L11 18 L5 22 Z" fill="#1d2b38" /><text x="11" y="36" textAnchor="middle" fontSize="11" fontFamily="var(--cw-font-serif)" fill="#1d2b38">N</text></svg>
      </div>
      <figcaption className="at-cap"><b>Plate 1.</b> Monterey Bay, {fmtDay(f.validFor, { year: true })}. Violet: NOAA C-HARM probability by 3 km cell (issued {fmtDay(f.issued)}, {f.freshness?.toLowerCase()}). Hatching: no model value. Amber: official limits. Green: shore stations.</figcaption>
    </figure>
  );
  return (
    <div className="at" data-mobile={mobile || undefined}>
      <header className="at-mast">
        <span className="at-brand">CoastWatch</span>
        <nav className="at-nav" aria-label="Primary"><a aria-current="page">Map</a><a>My Coast</a><a>Blooms</a><a>Fisheries</a></nav>
      </header>
      <StatusLine className="at-status" />
      <div className="at-body">
        <article className="at-col" tabIndex={0} aria-label="Reading column">
          <p className="at-dateline">{fmtDay(f.validFor, { weekday: true, year: true })} · Monterey Bay</p>
          <h1 className="at-head">{f.nCoverWord} notices cover the water off Santa Cruz. The nearest sample is {f.station.ageDays} days old.</h1>
          {mobile && plate}
          <p className="at-lede">At the point you chose<sup>1</sup>, {f.covering.map(noticeLabel).join(", ").replace(/, ([^,]*)$/, " and $1")} apply by their official limits.{f.limit && <> The {f.limitSpecies} {f.limitSharedBy > 1 ? "notices end" : "notice ends"} at {fmtLat(f.limit.lat)}<sup>4</sup>, {fmtKm(f.limit.km)} to the {f.limit.which}.</>}</p>
          <ol className="at-notes">
            <li value={2}><span className="mark mark-sample" aria-hidden="true" /><p><b>{f.station.station.name}</b>, {fmtKm(f.station.km)} {f.station.bearing}. Sampled {f.sampled}; particulate domoic acid {f.pDA?.toFixed(2)} ng/mL. A pier sample describes that pier on that day.</p></li>
            <li value={3}><span className="mark mark-field" aria-hidden="true" /><p><b>{f.cls ? `${f.cls.lo}–${f.cls.hi} % chance` : "No value"}</b> in this cell for {fmtDay(f.validFor)}: {f.threshold.toLowerCase()}. A forecast of the water, not a test of seafood.</p></li>
            <li value={4}><span className="mark mark-rule" aria-hidden="true" /><p><b>Official limit</b> {f.limitText}. Check which side you are on.</p></li>
          </ol>
          <section className="at-margin" aria-label="Margin notes">
            <p><span className="mark mark-gap" aria-hidden="true" /> <b>Not known.</b> {f.gaps.join(". ")}.</p>
            {f.disputed && <p><span className="mark mark-rule mark-rule-broken" aria-hidden="true" /> <b>Under review.</b> {f.disputed.by.agency} {f.disputed.by.release} appears to lift “{f.disputed.record.title}”; not shown as in effect.</p>}
          </section>
        </article>
        {!mobile && plate}
      </div>
    </div>
  );
}
