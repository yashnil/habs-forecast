import { useEffect, useRef, useState } from "react";
import { OfficialActions, ProbBar } from "../ui/Ledger";
import "./states.css";

/**
 * Loading, stale, unavailable and never-observed — the state vocabulary, and a live waterfall that
 * loads the real snapshot in CoastWatch's priority order under a chosen network profile.
 */
type Step = { key: string; label: string; priority: number; url?: string; local?: boolean };
const STEPS: Step[] = [
  { key: "frame", label: "Page structure, masthead, panel outline", priority: 1, local: true },
  { key: "place", label: "Chosen place and coordinates (from the link)", priority: 2, local: true },
  { key: "links", label: "CDFW and CDPH links and the shellfish line (bundled)", priority: 3, local: true },
  { key: "manifest", label: "Manifest: what exists, with its dates", priority: 4, url: "/data/v1/manifest.json" },
  { key: "official", label: "Official notices", priority: 4, url: "official" },
  { key: "ports", label: "Ports", priority: 4, url: "ports" },
  { key: "water", label: "Bundled coastline (map works without tiles)", priority: 5, url: "/basemap/water-fallback.json" },
  { key: "observations", label: "Shore samples", priority: 5, url: "observations" },
  { key: "charm", label: "Forecast field (one lead)", priority: 5, url: "charm" },
  { key: "relief", label: "Seafloor relief image", priority: 6, url: "/basemap/relief-sea-tint.webp" },
  { key: "satellite", label: "Satellite pixel chunk (only when asked)", priority: 6, url: "satellite" },
];
const PROFILES = { fast: [0, 0], slow: [600, 2900], "3g": [1500, 4500] } as const;

export function States() {
  const [profile, setProfile] = useState<keyof typeof PROFILES>("slow");
  const [rows, setRows] = useState<Record<string, { t0: number; t1?: number; ok?: boolean }>>({});
  const run = useRef(0);
  const start = async () => {
    const id = ++run.current;
    const T = performance.now();
    setRows({});
    const m = await fetch("/data/v1/manifest.json", { cache: "no-store" }).then((r) => r.json());
    const resolve = (u: string) => {
      if (u === "official") return `/data/v1/${m.official_url}`;
      if (u === "ports") return `/data/v1/${m.ports_url}`;
      if (u === "observations") return `/data/v1/${m.observations_url}`;
      if (u === "charm") return `/data/v1/${m.layers.find((l: { layer_id: string }) => l.layer_id === "charm_particulate_domoic_lead2").grid.url}`;
      if (u === "satellite") { const g = m.layers.find((l: { layer_id: string }) => l.layer_id === "olci300_chl_latest").grid; return `/data/v1/${g.chunks.url_template.replace("{row}", "3").replace("{col}", "1")}`; }
      return u;
    };
    STEPS.forEach((s, i) => {
      const t0 = performance.now() - T;
      setRows((r) => ({ ...r, [s.key]: { t0 } }));
      const [lo, span] = PROFILES[profile];
      const wait = s.local ? 0 : lo + ((i * 397) % (span || 1));
      const done = (ok: boolean) => run.current === id && setRows((r) => ({ ...r, [s.key]: { t0, t1: performance.now() - T, ok } }));
      if (s.local) { requestAnimationFrame(() => done(true)); return; }
      setTimeout(() => fetch(resolve(s.url!), { cache: "no-store" }).then((r) => done(r.ok), () => done(false)), wait);
    });
  };
  useEffect(() => { void start(); /* eslint-disable-next-line */ }, [profile]);
  const max = Math.max(1000, ...Object.values(rows).map((r) => r.t1 ?? 0));

  return (
    <main className="st">
      <header className="st-head">
        <a href="#/" className="t-meta">← Prototypes</a>
        <h1 className="t-display">Loading, gaps and failure</h1>
        <p className="t-lead">CoastWatch depends on live agency data. Loading should show what is coming, in a useful order, and never let a delay look like an absence or an absence look like a value.</p>
      </header>

      <section className="st-sec">
        <h2 className="t-title">The loading order</h2>
        <p className="t-body st-measure">Live: this page loads the real snapshot in priority order under a simulated network. Steps 1–3 are in the page itself and appear immediately — the official links never wait for data.</p>
        <div className="st-profiles" role="radiogroup" aria-label="Network profile">
          {(Object.keys(PROFILES) as (keyof typeof PROFILES)[]).map((p) => <button key={p} className="chip" role="radio" aria-checked={profile === p} onClick={() => setProfile(p)}>{p === "fast" ? "Fast" : p === "slow" ? "Slow" : "3G"}</button>)}
          <button className="btn btn-quiet" onClick={() => void start()}>Run again</button>
        </div>
        <ol className="wf">
          {STEPS.map((s) => {
            const r = rows[s.key];
            return (
              <li key={s.key} className="wf-row" data-p={s.priority}>
                <span className="wf-p t-micro">{s.priority}</span>
                <span className="wf-l t-label">{s.label}</span>
                <span className="wf-bar" aria-hidden="true">
                  {r && <span className="wf-fill" data-done={r.t1 != null} data-ok={r.ok} style={{ left: `${(r.t0 / max) * 100}%`, width: `${(((r.t1 ?? r.t0 + 60) - r.t0) / max) * 100}%` }} />}
                </span>
                <span className="wf-t t-micro num">{r?.t1 != null ? `${Math.round(r.t1)} ms` : r ? "…" : ""}</span>
              </li>
            );
          })}
        </ol>
        <p className="t-meta">Try it in the map: <a href="#/explorer?at=-122.06,36.80&show=forecast&net=3g&welcome=0">3G</a> · <a href="#/explorer?at=-122.06,36.80&show=forecast&fail=charm&welcome=0">forecast fails</a> · <a href="#/explorer?at=-122.06,36.80&show=forecast&fail=tiles&welcome=0">street map fails</a> · <a href="#/explorer?at=-122.06,36.80&fail=official,observations&welcome=0">notices and samples fail</a> · <a href="#/explorer?at=-122.06,36.80&fail=observations&cache=1&welcome=0">samples from device cache</a></p>
      </section>

      <section className="st-sec">
        <h2 className="t-title">Seven states, never confused</h2>
        <div className="st-grid">
          <Spec name="Loading" rule="The layout is already final; only the words are missing. A quiet bar, no shimmer. After 400 ms a line says what is being fetched.">
            <Row mark="mark-field" kind="Modelled" title={<span className="skel skel-line" style={{ width: "70%", display: "block" }}>.</span>} meta="Reading NOAA's forecast run…" />
          </Spec>
          <Spec name="Current" rule="The newest run or sample, inside its expected interval. Its own date is always shown — 'current' is never the only time information.">
            <Row mark="mark-field" kind="Modelled" title={<>60–70 % chance <span className="t-meta">in this 3 km cell</span></>} meta={<>C-HARM · for Mon, Oct 12 · <b>Current</b>, issued Oct 11</>} />
          </Spec>
          <Spec name="Not confirmed" rule="CoastWatch's last check of the source failed. Shows the last good run with its own dates and says a newer one may exist.">
            <Row mark="mark-field" kind="Modelled" title={<>60–70 % chance <span className="t-meta">in this 3 km cell</span></>} meta={<>C-HARM · for Mon, Oct 12 · <span className="fresh-not-confirmed">Not confirmed</span>, issued Oct 11</>} />
          </Spec>
          <Spec name="Stale or historical" rule="Older than the source's expected interval. Text and mark recede to grey; the value stays readable, the age leads.">
            <Row mark="mark-sample" kind="Measured at the shore" title="Monterey Wharf, 4.1 km southeast" meta="Toxin last measured Aug 31, 2022 · 4 years old" dim />
          </Spec>
          <Spec name="Unavailable" rule="The source failed now. Nothing is drawn in its place, the reason is one sentence, and everything else on the page still works.">
            <Row mark="mark-field" kind="Modelled" title="Forecast unavailable right now" meta="NOAA's run of Oct 11 exists, but CoastWatch could not load it just now" />
          </Spec>
          <Spec name="Never observed / no value" rule="A real gap in the data: hatched on the map, said in words, with the nearest real value shown where it is — never moved onto the point.">
            <Row mark="mark-gap" kind="Not known" title="No forecast for this nearshore cell" meta="Nearest cell with a value: 70–80 % chance, 2.9 km away" />
          </Spec>
          <Spec name="Saved copy" rule="Offline or failed, with a copy on this device: shown with the copy's own dates and the words 'saved copy from …'. Never relabelled as current.">
            <Row mark="mark-sample" kind="Measured at the shore · saved copy from Oct 12, 5:42 PM" title="Santa Cruz Wharf, 18 km north" meta="Last sampled Sep 30 · 12 days old" />
          </Spec>
        </div>
      </section>

      <section className="st-sec">
        <h2 className="t-title">When the scaffolding fails</h2>
        <ul className="st-rules">
          <li><b>Street map tiles fail.</b> The sea and coastline come from CoastWatch's bundled outline; data layers, notices and samples are unaffected. One note says so.</li>
          <li><b>Fonts fail.</b> The fallback stacks (Georgia, system UI) have similar widths; no text is hidden while fonts load (font-display: swap).</li>
          <li><b>Everything fails.</b> The page still shows the place, the official links and the phone line — the things that matter most for safety:</li>
        </ul>
        <div className="st-fallback"><OfficialActions /></div>
        <div className="st-probe"><p className="t-micro">Probability class bar (used in the ledger):</p><ProbBar value={0.64} /></div>
      </section>
    </main>
  );
}

function Spec({ name, rule, children }: { name: string; rule: string; children: React.ReactNode }) {
  return (
    <article className="spec">
      <h3 className="t-headline">{name}</h3>
      <div className="spec-demo">{children}</div>
      <p className="t-meta">{rule}</p>
    </article>
  );
}
function Row({ mark, kind, title, meta, dim }: { mark: string; kind: string; title: React.ReactNode; meta: React.ReactNode; dim?: boolean }) {
  return (
    <div className="lrow spec-row" data-dim={dim || undefined}>
      <div className="lrow-head" style={{ cursor: "default" }}>
        <span className={`mark ${mark}`} aria-hidden="true" />
        <span className="lrow-text">
          <span className="lrow-kind t-micro">{kind}</span>
          <span className="lrow-title">{title}</span>
          <span className="t-meta lrow-meta">{meta}</span>
        </span>
        <span />
      </div>
    </div>
  );
}
