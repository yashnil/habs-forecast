import { useMemo, useRef, useState } from "react";
import "./station.css";
import { useManifest, useObservations, useOfficial } from "../data/store";
import type { Manifest, Official } from "../data/types";
import { SeasonChart } from "../charts/SeasonChart";
import { Segmented } from "../charts/Segmented";
import type { ChartState } from "../charts/engine";
import {
  QUANTITIES, WINDOWS, type QuantityId, type WindowId, type StationWithModel,
  samplesOf, shortNotice, stationNotices, storySteps, windowDomain,
} from "../charts/season";
import { fmtValue } from "../charts/scale";
import { ageWords, fmtDay } from "../lib/time";

/**
 * Prototype 7 + 10 — "Reading a season": one station, one chart, and a short guided reading built
 * from the real samples and notices. Every step is a controlled, explainable chart transition.
 */
export function Station() {
  const man = useManifest();
  // the data hooks key on the manifest's file names, so they are only called once it is known
  return man.data ? <WithData m={man.data} /> : <StationView station={undefined} off={undefined} />;
}
function WithData({ m }: { m: Manifest }) {
  const id = new URLSearchParams(location.hash.split("?")[1] ?? "").get("id") ?? "HABs-SantaCruzWharf";
  const obs = useObservations(m);
  const off = useOfficial(m);
  const station = obs.data?.stations.find((s) => s.station_id === id) as StationWithModel | undefined;
  return <StationView station={station} off={off.data} />;
}

function StationView({ station, off }: { station: StationWithModel | undefined; off: Official | undefined }) {

  const [qid, setQ] = useState<QuantityId>("pDA");
  const [win, setWin] = useState<WindowId>("12m");
  const [showModel, setShowModel] = useState(false);
  const [stepIx, setStepIx] = useState<number | null>(null);
  const [table, setTable] = useState(false);
  const stepsRef = useRef<HTMLOListElement>(null);

  const samples = useMemo(() => (station ? samplesOf(station) : []), [station]);
  const notices = useMemo(() => (station ? stationNotices(off, station) : []), [off, station]);
  const steps = useMemo(() => storySteps(samples, notices), [samples, notices]);
  const q = QUANTITIES.find((x) => x.id === qid)!;
  const step = stepIx != null ? steps[stepIx] : null;

  const chartState = useMemo<ChartState | null>(() => {
    if (!samples.length) return null;
    const model = showModel && q.model && station?.charm?.history?.[q.model.key] ? station.charm.history[q.model.key] : null;
    return { samples, quantity: q, xDomain: windowDomain(samples, win), notices, model, highlight: step?.highlight ?? null };
  }, [samples, q, win, notices, showModel, station, step]);

  const goStep = (i: number) => {
    const s = steps[i];
    if (!s) return;
    setStepIx(i);
    setQ(s.quantity);
    setWin(s.window);
    requestAnimationFrame(() => {
      const li = stepsRef.current?.children[i] as HTMLElement | undefined;
      li?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest", inline: "center" });
      li?.querySelector("button")?.focus({ preventScroll: true });
    });
  };

  const last = station ? station.summaries.find((s) => s.variable === "pDA") : undefined;
  const lastSample = station?.sample_times.at(-1);
  const loading = !station;

  return (
    <main className="st-page">
      <nav className="st-crumbs t-meta" aria-label="Breadcrumb">
        <a href="#/">Prototypes</a> <span aria-hidden="true">/</span> Blooms <span aria-hidden="true">/</span> <span aria-current="page">{station?.name ?? "Station"}</span>
      </nav>

      <p className="st-status t-meta"><span className="mark mark-rule" aria-hidden="true" /> Official notices may be incomplete. Before harvesting or eating seafood, check <a className="ext" href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noopener">CDFW</a> and <a className="ext" href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noopener">CDPH</a>.</p>
      <header className="st-head">
        <div>
          <p className="st-kind t-label"><span className="mark mark-sample" aria-hidden="true" /> Shore station, measured about weekly</p>
          {loading ? <h1 className="t-display skel" style={{ width: "12ch" }}>Loading</h1> : <h1 className="t-display">{station.name}</h1>}
          <p className="t-lead st-lede">
            Water samples from this pier, tested for domoic acid, the toxin some algae make, and for the algae themselves. They describe the pier's water on the day of each sample.
          </p>
          {station && (
            <p className="t-coord st-coord">
              {station.lat.toFixed(3)}° N {Math.abs(station.lon).toFixed(3)}° W · {station.nearest_port_km} km from {station.nearest_port_name} · CalHABMAP
            </p>
          )}
        </div>
        <dl className="st-latest">
          <dt className="t-label">Latest sample</dt>
          <dd className="t-headline num">{lastSample ? fmtDay(lastSample, { year: true }) : <span className="skel">Sep 00, 2026</span>}</dd>
          <dd className="t-meta">{last?.last_date ? `${ageWords(last.last_date)} · ${last.n_last_365d} samples in 12 months` : " "}</dd>
          <dd><a className="t-label" href={`#/explorer?at=${station?.lon ?? -122.017},${station?.lat ?? 36.958}&select=station`}>View on the Ocean Map →</a></dd>
        </dl>
      </header>

      <div className="st-grid">
        <section className="st-chart-col" aria-labelledby="st-chart-h">
          <div className="st-controls">
            <h2 id="st-chart-h" className="sr-only">Samples over time</h2>
            <Segmented
              label="Quantity"
              value={qid}
              onChange={(v) => { setQ(v); setStepIx(null); }}
              options={QUANTITIES.map((x) => ({ id: x.id, label: x.label, short: x.tiny, hint: `${x.short} (${x.units})` }))}
            />
            <Segmented label="Time window" size="sm" value={win} onChange={(v) => { setWin(v); setStepIx(null); }} options={WINDOWS.map((w) => ({ id: w.id, label: w.label }))} />
          </div>

          <div className="st-axis-title t-meta">
            <span>{q.short}, <span className="num">{q.unitsWords}</span></span>
            {q.model && (
              <label className="st-model-toggle">
                <input type="checkbox" checked={showModel} onChange={(e) => setShowModel(e.target.checked)} />
                <span className="mark mark-field" aria-hidden="true" /> Show the forecast for nearby water
              </label>
            )}
          </div>

          <div className="st-chart-frame">
            {chartState ? <SeasonChart state={chartState} label={`${q.short} at ${station?.name}`} /> : <div className="skel" style={{ height: 300 }} />}
          </div>

          <ul className="st-legend t-meta" aria-label="What the marks mean">
            <li><span className="mark mark-sample" aria-hidden="true" /> A sample at this pier</li>
            {q.scale === "log" && <li><span className="st-hollow" aria-hidden="true" /> Reported 0 (not quantified)</li>}
            {notices.length > 0 && <li><span className="mark mark-rule" aria-hidden="true" /> When an official notice took effect</li>}
            <li><span className="mark mark-gap" aria-hidden="true" /> No samples for 3 weeks or more</li>
            {showModel && q.model && <li><span className="mark mark-field" aria-hidden="true" /> {q.model.label}, median of cells within {station?.charm?.radius_km ?? 15} km, right axis. A model probability, not a measurement.</li>}
          </ul>

          <div className="st-table-row">
            <button type="button" className="btn btn-quiet t-label" aria-expanded={table} onClick={() => setTable((t) => !t)}>
              {table ? "Hide table" : "Show as table"}
            </button>
            {station && <a className="t-label ext" href={station.source_url} target="_blank" rel="noopener noreferrer">Data source</a>}
          </div>
          {table && chartState && <SampleTable state={chartState} />}
        </section>

        <aside className="st-story" aria-labelledby="st-story-h">
          <h2 id="st-story-h" className="t-headline">Reading this season</h2>
          <p className="t-meta st-story-sub">{steps.length} steps, built from this station's samples and the official notices for these waters.</p>
          <ol
            ref={stepsRef}
            className="st-steps"
            onKeyDown={(e) => {
              if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); goStep(Math.min(steps.length - 1, (stepIx ?? -1) + 1)); }
              if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); goStep(Math.max(0, (stepIx ?? 1) - 1)); }
            }}
          >
            {loading && [0, 1, 2].map((i) => <li key={i} className="st-step"><div className="skel" style={{ height: 96 }} /></li>)}
            {steps.map((s, i) => (
              <li key={s.id} className="st-step">
                <button type="button" className="st-step-btn" aria-current={i === stepIx ? "step" : undefined} onClick={() => goStep(i)}>
                  <span className="st-step-n num" aria-hidden="true">{i + 1}</span>
                  <span className="st-step-title">{s.title}</span>
                  <span className="st-step-body">{s.body}</span>
                </button>
              </li>
            ))}
          </ol>
          <div className="st-story-nav">
            <button type="button" className="btn" disabled={stepIx == null || stepIx === 0} onClick={() => goStep((stepIx ?? 1) - 1)}>Previous</button>
            <button type="button" className="btn btn-primary" onClick={() => goStep(stepIx == null ? 0 : Math.min(steps.length - 1, stepIx + 1))} disabled={stepIx === steps.length - 1}>
              {stepIx == null ? "Start reading" : "Next"}
            </button>
            {stepIx != null && <button type="button" className="btn btn-quiet" onClick={() => setStepIx(null)}>Clear</button>}
          </div>
          {notices.length > 0 && (
            <p className="t-meta st-notices">
              Notices for these waters: {notices.map((r) => shortNotice(r)).join(", ")}. Check <a href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noopener noreferrer" className="ext">CDFW</a> and <a href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noopener noreferrer" className="ext">CDPH</a> before harvesting or eating seafood.
            </p>
          )}
        </aside>
      </div>
    </main>
  );
}

function SampleTable({ state }: { state: ChartState }) {
  const q = state.quantity;
  const rows = state.samples.filter((s) => s.t >= state.xDomain[0] && s.t <= state.xDomain[1]);
  return (
    <div className="st-table-wrap">
      <table className="st-table">
        <caption className="t-meta">{q.short} ({q.units}), {rows.length} samples in view, newest first</caption>
        <thead><tr><th scope="col">Date</th><th scope="col">{q.label} ({q.units})</th></tr></thead>
        <tbody>
          {[...rows].reverse().map((s) => {
            const v = s.v[q.id];
            return (
              <tr key={s.key}>
                <td className="num">{fmtDay(s.date, { year: true })}</td>
                <td className="num">{v == null ? "not measured" : v === 0 && q.scale === "log" ? "reported 0 (not quantified)" : fmtValue(v)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
