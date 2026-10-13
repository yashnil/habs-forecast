import "./system.css";

const ITEMS: { n: string; title: string; text: string; links: [string, string][] }[] = [
  { n: "1–6, 9", title: "Ocean Map", text: "Basemap-first map, the four-marks ledger, provenance threads, exact-cell forecast with three compositing modes, honest time bar, coastal journeys, purpose onboarding, research inspector and the mobile sheet.",
    links: [["Start here (first visit)", "#/explorer"], ["Water off Santa Cruz, forecast", "#/explorer?at=-122.06,36.80&show=forecast&welcome=0"], ["Lens around a point", "#/explorer?at=-122.06,36.80&show=forecast&field=lens&welcome=0"], ["Class lines", "#/explorer?at=-122.06,36.80&show=forecast&field=edges&welcome=0"], ["Nearshore gap explained", "#/explorer?at=-121.93,36.62&show=forecast&welcome=0"], ["Satellite pixel & cloud", "#/explorer?at=-122.06,36.80&show=satellite&welcome=0"], ["Research inspector", "#/explorer?at=-122.06,36.80&mode=research&show=forecast&welcome=0"]] },
  { n: "7, 10", title: "A season at Santa Cruz Wharf", text: "Chart transitions keyed to real samples, notices as rules on the timeline, gaps hatched, model on its own axis, and a five-step story computed from the data.", links: [["Open", "#/station"]] },
  { n: "8", title: "Loading, gaps and failure", text: "A live loading waterfall in priority order and the seven states. Every failure can be reproduced from a link.", links: [["Open", "#/states"], ["Map on 3G", "#/explorer?at=-122.06,36.80&show=forecast&net=3g&welcome=0"], ["Forecast and tiles fail", "#/explorer?at=-122.06,36.80&show=forecast&fail=charm,tiles&welcome=0"]] },
  { n: "Phase 2", title: "Three directions", text: "Marine Observatory, Living Coast and Editorial Atlas, on the same real state, with an evaluation.", links: [["Open", "#/directions"]] },
  { n: "Phase 3", title: "Design language", text: "Colour with measured contrast, type scale and motion tokens.", links: [["Open", "#/system"]] },
];

export function Index() {
  return (
    <main className="sys">
      <header className="sys-head">
        <p className="t-meta">CoastWatch · design prototypes · isolated from the live app</p>
        <h1 className="t-display">What is official, measured, modelled — and unknown</h1>
        <p className="t-lead">Interactive studies for CoastWatch's visual identity and interaction design. Real data from the published CoastWatch dataset of 13 October 2026 (00:42 UTC), shown on a fixed clock of 12 October 2026, 6 PM PDT. Not for decisions about fishing or eating seafood.</p>
      </header>
      <section className="sys-sec" style={{ gap: 0 }}>
        {ITEMS.map((it) => (
          <article key={it.title} className="idx-row">
            <span className="t-micro idx-n">{it.n}</span>
            <div>
              <h2 className="t-headline">{it.title}</h2>
              <p className="t-body" style={{ color: "var(--cw-ink-2)", maxWidth: 680 }}>{it.text}</p>
              <p className="idx-links">{it.links.map(([l, h]) => <a key={h} href={h} className="chip">{l}</a>)}</p>
            </div>
          </article>
        ))}
      </section>
      <style>{`.idx-row{display:grid;grid-template-columns:80px 1fr;gap:16px;padding:22px 0;border-bottom:1px solid var(--cw-hairline)}.idx-links{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}.idx-links .chip{text-decoration:none}@media(max-width:600px){.idx-row{grid-template-columns:1fr;gap:4px}}`}</style>
    </main>
  );
}
