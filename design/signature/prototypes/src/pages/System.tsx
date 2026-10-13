import { useEffect, useRef, useState } from "react";
import "./system.css";

/** The design language, live: colours with measured contrast, the type scale, motion tokens to play. */
const COLOURS: { group: string; items: [string, string, string?][] }[] = [
  { group: "Four marks on paper", items: [["--cw-official", "Official (text)", "--cw-page"], ["--cw-official-line", "Official rule line"], ["--cw-measured", "Measured", "--cw-page"], ["--cw-model", "Modelled", "--cw-page"], ["--cw-gap", "Not known", "--cw-page"], ["--cw-accent", "Context & interaction", "--cw-page"]] },
  { group: "Four marks on the dark sea", items: [["--cw-map-official", "Rule line", "--cw-sea"], ["--cw-map-measured", "Sample dot", "--cw-sea"], ["--cw-map-model", "Model line", "--cw-sea"], ["--cw-map-select", "Selection", "--cw-sea"], ["--cw-coastline", "Coastline", "--cw-sea"]] },
  { group: "Surfaces & ink", items: [["--cw-navy-900", "Frame"], ["--cw-page", "Paper"], ["--cw-surface", "Sheet"], ["--cw-surface-3", "Recessed"], ["--cw-ink", "Ink", "--cw-page"], ["--cw-ink-2", "Ink 2", "--cw-page"], ["--cw-ink-3", "Ink 3 (min. for text)", "--cw-page"]] },
  { group: "Sea", items: [["--cw-sea-shelf", "Shelf (tint)"], ["--cw-sea", "Sea"], ["--cw-sea-deep", "Abyss"], ["--cw-land", "Land"]] },
  { group: "Forecast classes (published palette, unchanged)", items: Array.from({ length: 10 }, (_, i) => [`--cw-p${i}`, `${i * 10}–${i * 10 + 10} %`] as [string, string]) },
];
const TYPE: [string, string, string][] = [
  ["t-display", "Display · Newsreader 600, opsz 72", "Waters off Santa Cruz"],
  ["t-title", "Title · Newsreader 500, opsz 48", "Off Monterey"],
  ["t-headline", "Headline · Newsreader 500, opsz 24", "Three notices cover these waters"],
  ["t-lead", "Lead · Plex Sans 400 17/26", "The nearest shore sample is 12 days old, 18 km north."],
  ["t-body", "Body · Plex Sans 400 15/23", "A pier sample describes the water at that pier on that day. It is not a measurement of seafood."],
  ["t-ui", "UI · Plex Sans 500 14", "Around my point"],
  ["t-label", "Label · Plex Sans 500 13", "CDPH advisories"],
  ["t-meta", "Meta · Plex Sans 400 13", "Last sampled Sep 30 · 12 days old"],
  ["t-micro", "Micro · Plex Sans 500 12 (axes, attributions)", "0 %  ·  50 %  ·  100 %"],
  ["t-value", "Value · Plex Sans 500 28, tabular", "0.20"],
  ["t-coord", "Coordinate · Plex Mono 400 12", "36°48.00′ N  122°03.60′ W"],
];
const MOTION: [string, string, string, string][] = [
  ["--cw-d-micro", "--cw-ease-standard", "Press, hover, toggle", "110 ms"],
  ["--cw-d-short", "--cw-ease-standard", "Tooltips, chips, small reveals", "180 ms"],
  ["--cw-d-medium", "--cw-ease-settle", "Sheets settling, selection marker", "280 ms"],
  ["--cw-d-long", "--cw-ease-standard", "Chart rescale, arrangement", "460 ms"],
  ["--cw-d-swap", "--cw-ease-linear", "Data swap: out, then in — never blended", "100 + 140 ms"],
  ["--cw-d-thread", "--cw-ease-standard", "Provenance threads drawing out (once)", "520 ms"],
];

function lum(hex: string) {
  const m = hex.trim().match(/^#([0-9a-f]{6})$/i);
  if (!m) return null;
  const n = parseInt(m[1], 16);
  const c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => { const s = v / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
function ratio(a: string, b: string) {
  const la = lum(a), lb = lum(b);
  if (la == null || lb == null) return null;
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

export function System() {
  const [vals, setVals] = useState<Record<string, string>>({});
  useEffect(() => {
    const cs = getComputedStyle(document.documentElement);
    const all = COLOURS.flatMap((g) => g.items.flatMap(([a, , b]) => [a, b].filter(Boolean) as string[]));
    setVals(Object.fromEntries(all.map((k) => [k, cs.getPropertyValue(k).trim()])));
  }, []);
  return (
    <main className="sys">
      <header className="sys-head">
        <a href="#/" className="t-meta">← Prototypes</a>
        <h1 className="t-display">CoastWatch design language</h1>
        <p className="t-lead">Every piece of information is drawn with the mark that matches how it is known — a <b>line</b> for an official rule, a <b>point</b> for a sample, a <b>field</b> of cells for a model, <b>hatching</b> for what isn't known. The same mark appears on the map, in sentences and in charts.</p>
        <div className="sys-marks">
          {[["mark-rule", "Official", "A line. Latitude limits, area outlines. Amber."], ["mark-sample", "Measured", "A point that carries its date: filled when recent, a ring when old."], ["mark-pixel", "Seen by satellite", "A pixel with its own date. Never smoothed."], ["mark-field", "Modelled", "A field of cells, exact edges. Violet. Never a point."], ["mark-gap", "Not known", "Hatching in grey. Never a colour from a data palette."]].map(([m, l, d]) => (
            <div key={m} className="sys-mark"><span className={`mark ${m}`} aria-hidden="true" /><b className="t-ui">{l}</b><span className="t-meta">{d}</span></div>
          ))}
        </div>
      </header>

      <section className="sys-sec">
        <h2 className="t-title">Colour</h2>
        {COLOURS.map((g) => (
          <div key={g.group} className="sys-group">
            <h3 className="t-label">{g.group}</h3>
            <div className="sw-grid">
              {g.items.map(([v, label, on]) => {
                const r = on && vals[v] && vals[on] ? ratio(vals[v], vals[on]) : null;
                return (
                  <figure key={v} className="sw">
                    <span className="sw-chip" style={{ background: `var(${v})` }}>{on && <span style={{ color: `var(${v})`, background: `var(${on})` }} className="sw-on">Aa</span>}</span>
                    <figcaption><span className="t-label">{label}</span><span className="t-coord">{v} {vals[v]}</span>{r && <span className={`t-micro ${r >= 4.5 ? "pass" : r >= 3 ? "large" : "fail"}`}>{r.toFixed(1)}:1 {r >= 4.5 ? "AA text" : r >= 3 ? "AA large / graphics" : "decorative only"}</span>}</figcaption>
                  </figure>
                );
              })}
            </div>
          </div>
        ))}
        <p className="t-meta sys-note">No red, green or amber implies seafood safety. Amber is reserved for official information (it means "an agency said this"), green-teal for measurements, violet for model output. There is no "all clear" colour anywhere in the system.</p>
      </section>

      <section className="sys-sec">
        <h2 className="t-title">Type</h2>
        <p className="t-body sys-note">Two families, used more strictly: Newsreader (variable, optical size) only for place names and headlines that <i>state an answer</i>; IBM Plex Sans for everything read or operated; Plex Mono only for coordinates and identifiers. Numbers are tabular Plex Sans, never mono. Nothing below 12 px; sentence case throughout (no uppercase eyebrows).</p>
        <div className="type-list">
          {TYPE.map(([cls, spec, sample]) => <div key={cls} className="type-row"><span className="t-micro type-spec">{spec}</span><span className={cls}>{sample}</span></div>)}
        </div>
      </section>

      <section className="sys-sec">
        <h2 className="t-title">Motion</h2>
        <p className="t-body sys-note">Motion explains change of place, state or attention. It never animates data between two values, never loops without a reason, and under reduced motion every spatial move becomes a 120 ms fade.</p>
        <div className="mo-list">{MOTION.map(([d, e, use, ms]) => <MotionRow key={d} d={d} e={e} use={use} ms={ms} />)}</div>
      </section>
    </main>
  );
}

function MotionRow({ d, e, use, ms }: { d: string; e: string; use: string; ms: string }) {
  const box = useRef<HTMLSpanElement>(null);
  const [on, setOn] = useState(false);
  return (
    <div className="mo-row">
      <button className="btn" onClick={() => setOn(!on)} aria-pressed={on}>Play</button>
      <span className="mo-track"><span ref={box} className="mo-dot" style={{ transitionDuration: `var(${d})`, transitionTimingFunction: `var(${e})`, transform: on ? "translateX(calc(var(--track) - 18px))" : "none", opacity: d === "--cw-d-swap" ? (on ? 1 : 0.25) : 1 }} /></span>
      <span className="t-label">{use}</span>
      <span className="t-coord">{d} · {e} · {ms}</span>
    </div>
  );
}
