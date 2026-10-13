import { useId, useState } from "react";
import { OFFICIAL_LINKS } from "../content/links";
import type { Res } from "../data/store";
import type { Manifest } from "../data/types";
import { fmtKm } from "../lib/geo";
import type { NoticeGeo, Reading } from "../lib/provenance";
import { ageWords, fmtDay, forecastFreshness, relDay } from "../lib/time";
import type { ThreadKind } from "../map/threads";

/**
 * The four-marks ledger: what is known about the chosen water, one row per *kind* of knowing, in a
 * fixed order (official → measured → model → not known). Level 1 is the row's one-line answer;
 * Level 2 opens the row; Level 3 ("the working") shows exact values, cells and sources. Hovering
 * or focusing a row brightens its provenance thread on the map.
 */
export type RowKind = "official" | "measured" | "satellite" | "model" | "unknown";
const THREAD: Record<RowKind, ThreadKind | null> = { official: "notice", measured: "station", satellite: "satellite", model: "model", unknown: null };

export function Ledger(props: {
  reading: Reading | null;
  official: Res<unknown>;
  observations: Res<unknown>;
  forecast: Res<unknown> & { failed?: boolean };
  manifest?: Manifest;
  research: boolean;
  species: string | null;
  onThread: (k: ThreadKind | null) => void;
  satelliteOn: boolean;
}) {
  const { reading: r, research } = props;
  const [open, setOpen] = useState<RowKind | null>(research ? "model" : null);
  const saved = (k: RowKind) => ((k === "official" ? props.official : k === "measured" ? props.observations : k === "model" ? props.forecast : null)?.status === "cached" ? "saved copy on this device" : undefined);
  const row = (kind: RowKind, mark: string, title: React.ReactNode, meta: React.ReactNode, body: React.ReactNode, state: "ready" | "loading" | "failed" = "ready") => (
    <LedgerRow key={kind} kind={kind} note={saved(kind)} mark={mark} title={title} meta={meta} state={state} open={open === kind || research}
      onToggle={() => setOpen(open === kind ? null : kind)}
      onHover={(on) => props.onThread(on ? THREAD[kind] : null)}>
      {body}
    </LedgerRow>
  );

  /* ------------------------------------------------ official */
  const noticesHere = (r?.notices ?? []).filter((n) => n.tier !== "outside" && matchesSpecies(n, props.species));
  const applies = noticesHere.filter((n) => n.tier === "geographic");
  const may = noticesHere.filter((n) => n.tier === "unresolved");
  const nearbyOutside = (r?.notices ?? []).filter((n) => n.tier === "outside" && n.nearestLimit && n.nearestLimit.km < 40 && matchesSpecies(n, props.species));
  const offState = props.official.status === "loading" ? "loading" : props.official.status === "failed" ? "failed" : "ready";
  const officialTitle = offState === "failed" ? "Notices could not be loaded" : applies.length
    ? `${applies.length} ${applies.length === 1 ? "notice covers" : "notices cover"} these waters${props.species ? ` for ${props.species}` : ""}`
    : may.length ? `${may.length} ${may.length === 1 ? "notice" : "notices"} may cover these waters` : "No notice's official limits include these waters";
  const officialMeta = offState === "failed" ? "Check the agencies directly" : applies.length || may.length ? "From CDFW and CDPH · not verified by CoastWatch" : "That is not the same as safe or open";

  /* ------------------------------------------------ measured (station) */
  const st = r?.station;
  const obsState = props.observations.status === "loading" ? "loading" : props.observations.status === "failed" ? "failed" : "ready";
  const measuredTitle = obsState === "failed" ? "Shore samples could not be loaded" : !st ? "…" : st.tooFar
    ? `No shore sampling within 25 km`
    : `${st.station.name}, ${fmtKm(st.km)} ${st.bearing}`;
  const measuredMeta = !st ? "" : st.tooFar
    ? `Nearest: ${st.station.name}, ${fmtKm(st.km)} away — too far to describe these waters`
    : st.lastDate ? `Last sampled ${fmtDay(st.lastDate)} · ${ageWords(st.lastDate)}${st.toxinDate && st.toxinDate !== st.lastDate ? ` · toxin last measured ${fmtDay(st.toxinDate, { year: true })}` : ""}` : "No samples";

  /* ------------------------------------------------ model */
  const m = r?.model;
  const run = props.manifest?.forecast_runs.find((x) => x.group_id === "charm");
  const charmSrc = props.manifest?.sources.find((s) => s.source_id === "charm");
  const fresh = run ? forecastFreshness(run.issued_date, charmSrc) : null;
  const modelState = props.forecast.failed ? "failed" : !m || m.status === "loading" ? "loading" : "ready";
  const modelTitle = props.forecast.failed ? "Forecast unavailable right now"
    : m?.status === "value" ? <>{probWords(m.value)} <span className="t-meta">in this 3 km cell</span></>
    : m?.status === "gap" ? gapTitle(m.reason.kind) : "…";
  const modelMeta = props.forecast.failed ? <>NOAA's run of {run ? fmtDay(run.issued_date) : "—"} exists, but CoastWatch could not load it just now</> : m && m.status !== "loading" && m.layer ? <>
    C-HARM · for {fmtDay(m.layer.time.valid_date!, { weekday: true })} · <span className={`fresh fresh-${(fresh ?? "").replace(" ", "-").toLowerCase()}`}>{fresh}</span>, issued {fmtDay(m.layer.time.issued_date!)}
  </> : null;

  /* ------------------------------------------------ satellite */
  const sat = r?.satellite;
  const satTitle = !sat || sat.status === "loading" ? "…" : sat.status === "value" ? <>{fmtChl(sat.value)} <span className="t-meta">mg/m³ chlorophyll</span></> : sat.status === "gap" ? "No clear view this week" : "";
  const satMeta = sat?.status === "value" ? `Clear view ${fmtDay(sat.date)} (${relDay(sat.date)}) · Sentinel-3 300 m pixel` : sat?.status === "gap" && sat.reason.kind === "cloud" && sat.reason.nearest ? `Nearest clear pixel ${fmtKm(sat.reason.nearest.km)} away, ${fmtDay(sat.reason.nearest.date)}` : "Cloud, fog or no overpass";

  /* ------------------------------------------------ not known */
  const unknown: string[] = [];
  if (m?.status === "gap") unknown.push(m.reason.kind === "nearshore-mask" ? "No toxin forecast for this nearshore cell" : m.reason.kind === "outside-domain" ? "Outside the forecast model's area" : "No forecast right now");
  if (st && (st.tooFar || (st.ageDays ?? 99) > 14)) unknown.push(st.tooFar ? "No recent water samples nearby" : `Nothing measured at ${st.station.name} since ${fmtDay(st.lastDate!)}`);
  if (props.satelliteOn && sat?.status === "gap") unknown.push("No clear satellite view this week");
  unknown.push("Toxin in seafood — only CDPH testing tells you that, and CoastWatch does not have those results");

  return (
    <div className="ledger" data-level={research ? 3 : undefined}>
      {r?.disputed.length ? (
        <div className="ledger-disputed enter">
          <span className="mark mark-rule mark-rule-broken" aria-hidden="true" />
          <div>
            <p className="t-label"><b>{r.disputed[0].by.agency} release {r.disputed[0].by.release}</b> appears to lift “{r.disputed[0].record.title}”. CoastWatch has not reviewed it yet, so it is not shown as in effect.{" "}
              <a href={r.disputed[0].by.url} target="_blank" rel="noopener" className="ext">Read the release</a></p>
            <p className="t-micro">Prototype fixture mirroring PR #40</p>
          </div>
        </div>
      ) : null}
      {row("official", "mark-rule", officialTitle, officialMeta, (
        <div className="ledger-body">
          {[...applies, ...may].map((n) => <NoticeLine key={n.record.id} n={n} research={research} />)}
          {!applies.length && !may.length && <p className="t-meta">Statewide and nearby notices are in the full list. A missing notice does not mean an area is open or that seafood is safe.</p>}
          {nearbyOutside.length > 0 && (
            <p className="t-meta">Nearby but outside its official limits: {nearbyOutside.map((n) => `${n.record.agency} ${shortTitle(n.record.title)} (limit ${fmtKm(n.nearestLimit!.km)} ${n.nearestLimit!.which === "south" ? "north" : "south"} of here)`).join("; ")}.</p>
          )}
          <OfficialActions />
        </div>
      ), offState)}
      {row("measured", "mark-sample", measuredTitle, measuredMeta, st && (
        <div className="ledger-body">
          {!st.tooFar && st.lastValue != null && st.toxinDate && (
            <p><span className="t-value-sm">{st.lastValue.toFixed(2)}</span><span className="t-unit">ng/mL toxin in seawater (particulate domoic acid), {fmtDay(st.toxinDate, { year: st.toxinDate.slice(0, 4) !== st.lastDate?.slice(0, 4) })}</span></p>
          )}
          <p className="t-meta">A pier sample describes the water at that pier on that day. It is not a measurement of seafood, and it does not describe these waters.</p>
          {research && <dl className="ledger-dl"><dt>Station</dt><dd className="t-coord">{st.station.station_id}</dd><dt>Position</dt><dd className="t-coord">{st.station.lat.toFixed(4)}°, {st.station.lon.toFixed(4)}°</dd><dt>Distance</dt><dd className="num">{st.km.toFixed(2)} km {st.bearing}</dd><dt>Source</dt><dd><a href={st.station.source_url} target="_blank" rel="noopener" className="ext">SCCOOS ERDDAP</a></dd></dl>}
          <a className="btn btn-quiet" href={`#/station?id=${st.station.station_id}`}>The season at {st.station.name} →</a>
        </div>
      ), obsState)}
      {row("model", "mark-field", modelTitle, modelMeta, m && m.status !== "loading" && (
        <div className="ledger-body">
          {m.status === "value" && <>
            <ProbBar value={m.value} />
            <p className="t-meta">{m.layer.threshold_text}. A forecast of the water, not a test of seafood. A low chance does not mean safe.</p>
            {charmSrc?.outcome !== "ok" && <p className="t-meta"><span className="fresh fresh-not-confirmed">Not confirmed</span> — CoastWatch could not reach NOAA at its last check; a newer run may exist.</p>}
          </>}
          {m.status === "gap" && <GapExplainer reason={m.reason} />}
          {research && m.status === "value" && (
            <dl className="ledger-dl">
              <dt>Cell value</dt><dd className="num">{m.value.toFixed(4)} <span className="t-meta">± 0.00001 (quantization)</span></dd>
              <dt>Cell centre</dt><dd className="t-coord">{m.center[1].toFixed(3)}°, {m.center[0].toFixed(3)}°</dd>
              <dt>Cell edges</dt><dd className="t-coord">{m.cell.map((v) => v.toFixed(3)).join(", ")}</dd>
              <dt>Layer</dt><dd className="t-coord">{m.layer.layer_id}</dd>
              <dt>Issue date</dt><dd>{m.layer.time.issued_date} <span className="t-meta">(inferred: nowcast day + 1)</span></dd>
            </dl>
          )}
        </div>
      ), modelState)}
      {props.satelliteOn && row("satellite", "mark-pixel", satTitle, satMeta, sat && sat.status !== "loading" && sat.status !== "off" && (
        <div className="ledger-body">
          <p className="t-meta">Chlorophyll is algae biomass seen from space. It is not toxin and does not identify Pseudo-nitzschia. Each pixel keeps its own date.</p>
          {sat.status === "gap" && <GapExplainer reason={sat.reason} />}
        </div>
      ), !sat || sat.status === "loading" ? "loading" : "ready")}
      {row("unknown", "mark-gap", "What isn't known here", `${unknown.length} ${unknown.length === 1 ? "gap" : "gaps"}`, (
        <ul className="ledger-gaps">{unknown.map((u) => <li key={u}>{u}</li>)}</ul>
      ))}
    </div>
  );
}

function LedgerRow({ kind, mark, title, meta, state, open, onToggle, onHover, children, note }: {
  kind: RowKind; note?: string; mark: string; title: React.ReactNode; meta: React.ReactNode; state: "ready" | "loading" | "failed"; open: boolean;
  onToggle: () => void; onHover: (on: boolean) => void; children: React.ReactNode;
}) {
  const id = useId();
  return (
    <section className="lrow" data-kind={kind} data-state={state} data-open={open} onPointerEnter={() => onHover(true)} onPointerLeave={() => onHover(false)}>
      <button className="lrow-head" aria-expanded={open} aria-controls={id} onClick={onToggle} onFocus={() => onHover(true)} onBlur={() => onHover(false)}>
        <span className={`mark ${mark}`} aria-hidden="true" />
        <span className="lrow-text">
          <span className="lrow-kind t-micro">{KIND_WORD[kind]}{note && <span className="lrow-note"> · {note}</span>}</span>
          {state === "loading" ? <span className="lrow-title"><span className="skel skel-line" style={{ width: "72%", display: "block" }}>Loading</span></span>
            : <span className="lrow-title">{title}</span>}
          {state === "loading" ? <span className="t-meta lrow-loading">{LOADING_WORD[kind]}</span> : meta ? <span className="t-meta lrow-meta">{meta}</span> : null}
        </span>
        <span className="lrow-chev" aria-hidden="true" />
      </button>
      <div className="reveal" data-open={open} id={id}>
        <div>{state === "ready" ? children : state === "failed" ? <div className="ledger-body"><p className="t-meta">This part could not be loaded. Everything else on this page is unaffected.</p><OfficialActions /></div> : null}</div>
      </div>
    </section>
  );
}
const KIND_WORD: Record<RowKind, string> = { official: "Official", measured: "Measured at the shore", satellite: "Seen by satellite", model: "Modelled", unknown: "Not known" };
const LOADING_WORD: Record<RowKind, string> = { official: "Reading the notice list…", measured: "Finding the nearest shore samples…", satellite: "Reading the satellite pixel…", model: "Reading NOAA's forecast run…", unknown: "" };

function NoticeLine({ n, research }: { n: NoticeGeo; research: boolean }) {
  const r = n.record;
  return (
    <article className="notice">
      <p className="t-micro notice-agency">{r.agency} · {ACTION[r.action] ?? r.action}{n.tier === "unresolved" ? " · may apply" : ""}</p>
      <h4 className="notice-title">{r.title}</h4>
      <p className="t-meta">
        {r.effective_date && <>Since {fmtDay(r.effective_date, { weekday: true })} · </>}
        {r.sources[0] && <a href={r.sources[0].url} target="_blank" rel="noopener" className="ext">{r.sources[0].label}</a>}
      </p>
      {n.kind === "lat-band" && n.nearestLimit && (
        <p className="notice-limit t-meta"><span className="mark mark-rule" aria-hidden="true" /> Its {n.nearestLimit.which} limit is {fmtKm(n.nearestLimit.km)} {n.nearestLimit.which === "south" ? "south" : "north"} of this point. Check which side you are on.</p>
      )}
      {n.tier === "unresolved" && <p className="t-meta">The notice names an area on land and does not say how far offshore it reaches.</p>}
      {research && <p className="t-coord">{r.id}</p>}
    </article>
  );
}
const ACTION: Record<string, string> = { quarantine: "Quarantine", consumption_advisory: "Health advisory", take_restriction: "Take restriction", fishery_closure: "Fishery closure", special_advisory: "Special advisory" };

export function OfficialActions() {
  return (
    <div className="official-actions">
      <a className="btn" href={OFFICIAL_LINKS.cdph.url} target="_blank" rel="noopener"><span className="mark mark-rule" aria-hidden="true" />CDPH advisories</a>
      <a className="btn" href={OFFICIAL_LINKS.cdfw.url} target="_blank" rel="noopener"><span className="mark mark-rule" aria-hidden="true" />CDFW advisories</a>
      <a className="btn btn-quiet" href={`tel:${OFFICIAL_LINKS.phone.tel}`}>Call {OFFICIAL_LINKS.phone.text}</a>
    </div>
  );
}

function GapExplainer({ reason }: { reason: import("../lib/provenance").GapReason }) {
  const near = "nearest" in reason ? reason.nearest : undefined;
  return (
    <div className="gapx">
      <p className="t-label"><span className="mark mark-gap" aria-hidden="true" /> {GAP_WHY[reason.kind]}</p>
      <p className="t-meta">{reason.detail}</p>
      {near && "date" in near ? (
        <p className="t-meta">Nearest clear pixel: {fmtChl(near.value)} mg/m³, {fmtKm(near.km)} away, seen {fmtDay(near.date as string)}. Shown on the map; it describes that pixel, not this one.</p>
      ) : near ? (
        <p className="t-meta">Nearest cell with a value: {probWords(near.value)}, {fmtKm(near.km)} away. Shown on the map; it describes that cell, not this one.</p>
      ) : null}
    </div>
  );
}
const GAP_WHY: Record<string, string> = {
  "outside-domain": "Outside the model's area",
  "nearshore-mask": "A nearshore cell the model leaves out",
  cloud: "Clouds or fog hid the water",
  "no-sampling": "No sampling nearby",
  "too-old": "The last measurement is too old",
  "source-unavailable": "The source could not be reached",
  land: "That's land",
};
function gapTitle(k: string) { return k === "nearshore-mask" ? "No forecast for this nearshore cell" : k === "outside-domain" ? "Outside the forecast model's area" : k === "source-unavailable" ? "Forecast unavailable right now" : "No forecast here"; }

/** "70–80 % chance" — the display class, with the exact value one level down. */
export function probWords(v: number) {
  const c = Math.min(9, Math.floor(v * 10 + 1e-9));
  return `${c * 10}–${c * 10 + 10} % chance`;
}
export function ProbBar({ value }: { value: number }) {
  const c = Math.min(9, Math.floor(value * 10 + 1e-9));
  return (
    <div className="probbar" role="img" aria-label={`Probability class ${c * 10} to ${c * 10 + 10} percent`}>
      {Array.from({ length: 10 }, (_, i) => <span key={i} data-on={i === c} style={{ background: `var(--cw-p${i})` }} />)}
      <span className="probbar-scale t-micro"><span>0 %</span><span>50 %</span><span>100 %</span></span>
    </div>
  );
}
const fmtChl = (v: number) => (v < 1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : v.toFixed(0));
function shortTitle(t: string) { return t.length > 48 ? t.slice(0, 46) + "…" : t; }
function matchesSpecies(n: NoticeGeo, sp: string | null) {
  if (!sp) return true;
  return n.record.species.some((s) => s.toLowerCase().includes(sp.toLowerCase()) || sp.toLowerCase().includes(s.toLowerCase()));
}
