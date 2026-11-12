"use client";

import { FollowLine } from "@/nx/alerts/FollowLine";
import { AnimatePresence, motion, MotionConfig } from "motion/react";
import { useEffect, useId, useMemo, useState } from "react";
import type { FisheryId } from "@/pilot/contracts";
import { fisheryLabel as labelOf } from "@/pilot/facts";
import { readSaved } from "@/pilot/today/prefs";
import { saveTodayOffline } from "@/pilot/today/offline";
import { ChartRegulation, RAMP_FISHERIES, rampZoneAt } from "@/nx/chart/regulation";
import { ChartScene } from "@/nx/chart/scene";
import { Agenda, type AgendaItem } from "@/nx/ds/Agenda";
import { BigDate } from "@/nx/ds/BigDate";
import { Button } from "@/nx/ds/Button";
import { dayTime } from "@/nx/ds/format";
import { markFor } from "@/nx/ds/kinds";
import { Legend } from "@/nx/ds/Mark";
import { MarginRow, MarginRows } from "@/nx/ds/Margin";
import { pan, register, registerFrom, registerTo, settle } from "@/nx/ds/motion";
import { ProvenanceSheet, Stamp, type Provenance } from "@/nx/ds/Stamp";
import { Loading, SourcesDiffer, Unavailable } from "@/nx/ds/States";
import { Statement } from "@/nx/ds/Statement";
import { useMedia, usePhone } from "@/nx/ds/useMedia";
import { CANYON_AXIS } from "@/nx/chart/features";
import { distanceKm, phoneBounds, portWater, todayBounds } from "./camera";
import type { ChangeLine } from "./change";
import { useChangeLine } from "./useChangeLine";
import { conditionsUrl, useConditions, useTodayInputs, type Initial } from "./data";
import { FisheryChooser } from "./FisheryChooser";
import { PORT_NAME_LAYOUT } from "./layout";
import { Advisories, GateList, type GateItem } from "@/nx/ds/GateList";
import { gateHrefs, type Gate } from "./gates";
import { buildToday, othersLine, scopeWords, type TodayInputs, type TodayView as View } from "./model";
import { COMPACT, TodaySoundings } from "./Soundings";
import { chooseFishery, useFisheries, useTourState } from "./store";
import { Tour, TourLink } from "./Tour";
import s from "./today.module.css";

export type TodayProps = {
  port: { key: string; name: string; lon: number; lat: number } | null;
  initial: Initial;
  /** first visit at `/`: the chart stays on the coast until a fishery is chosen */
  arriving?: boolean;
};


const canyonHead = { lon: CANYON_AXIS[0][0], lat: CANYON_AXIS[0][1] };

/**
 * Today at your port (journeys.md Journeys 1 and 2): the season status for your fishery, the next date that
 * can change it, what changed since your last visit, the dated agenda, two soundings on the chart and the
 * rest as one-line rows. Each value opens its provenance. Every missing source says so in words.
 */
export function TodayView({ port, initial, arriving = false }: TodayProps) {
  const fisheries = useFisheries();
  const data = useTodayInputs(port?.key ?? null, initial);
  const primary = fisheries?.[0];
  if (!port) return null;
  const inputs = data.state === "ok" ? data.inputs : null;
  // RAMP's zone lines on the chart for a crab reader, the port's own zone heavier (E T43)
  const ramp = !!fisheries?.some((f) => RAMP_FISHERIES.has(f));
  return (
    <MotionConfig reducedMotion="user">
      <ChartRegulation spec={ramp ? { zones: { highlight: rampZoneAt(port.lat) } } : null} />
      <section className={s.today} data-testid="nx-today" aria-labelledby="nx-today-title">
        {fisheries === undefined ? (
          <PortHeading name={port.name} />
        ) : !primary ? (
          <>
            {!arriving ? <ChartScene scene={{ id: `today:${port.key}`, view: { bounds: portWater(port) }, focus: { key: port.key, name: port.name, lon: port.lon, lat: port.lat } }} /> : null}
            <FisheryChooser port={port} inputs={data.state === "loading" ? undefined : inputs} onChoose={chooseFishery} />
          </>
        ) : data.state === "loading" ? (
          <>
            <ChartScene scene={{ id: `today:${port.key}`, view: { bounds: portWater(port) }, focus: { key: port.key, name: port.name, lon: port.lon, lat: port.lat } }} />
            <PortHeading name={port.name} fishery={primary} />
            <Loading>Reading the records for {port.name}</Loading>
          </>
        ) : data.state === "error" || !inputs ? (
          <>
            <ChartScene scene={{ id: `today:${port.key}`, view: { bounds: portWater(port) }, focus: { key: port.key, name: port.name, lon: port.lon, lat: port.lat } }} />
            <PortHeading name={port.name} fishery={primary} />
            <Unavailable
              what={`Today at ${port.name} could not be read`}
              why="Neritic's published records did not load. The agencies' own pages still work."
              next={
                <p className={s.links}>
                  <Button variant="text" href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" external>
                    CDFW advisories
                  </Button>
                  <Button variant="text" href="https://www.weather.gov/mtr/" external>
                    NWS marine forecast
                  </Button>
                </p>
              }
            />
          </>
        ) : (
          <Assembled port={port} inputs={inputs} fishery={primary} />
        )}
      </section>
    </MotionConfig>
  );
}

function PortHeading({ name, fishery }: { name: string; fishery?: FisheryId }) {
  return (
    <p className={s.scope}>
      <motion.span layoutId={PORT_NAME_LAYOUT} className={s.portName} transition={pan}>
        {name}
      </motion.span>
      {fishery ? <span className={s.scopeFishery}> · {scopeWords(labelOf(fishery))}</span> : null}
    </p>
  );
}

/**
 * After the page has drawn, a copy for offline use (sw.js answers `/` with it), at most twice a day. Its time is
 * the page's clock (`now`, the server's, which a fixture preview pins), never the device's, so a saved copy is
 * never stamped after the moment the page shows.
 */
function useOfflineCopy(key: string, name: string, url: string, ready: boolean, now: string) {
  const [at, setAt] = useState<string | null>(null);
  useEffect(() => {
    const saved = readSaved().find((x) => x.place === key);
    if (saved) setAt(saved.at);
    if (!ready || process.env.NODE_ENV !== "production") return;
    if (saved && Math.abs(Date.parse(now) - Date.parse(saved.at)) < 12 * 3_600_000) return;
    const idle = (window as unknown as { requestIdleCallback?: (f: () => void) => number }).requestIdleCallback ?? ((f: () => void) => window.setTimeout(f, 2500));
    let live = true;
    idle(() => {
      if (!live || (window as unknown as { __NERITIC_SAVED__?: unknown }).__NERITIC_SAVED__) return;
      void saveTodayOffline(key, name, url, now).then((r) => live && r.ok && setAt(r.at));
    });
    return () => {
      live = false;
    };
  }, [key, name, url, ready, now]);
  return at;
}

function Assembled({ port, inputs, fishery }: { port: NonNullable<TodayProps["port"]>; inputs: TodayInputs; fishery: FisheryId }) {
  const curl = conditionsUrl(inputs.place.lat, inputs.place.lon);
  const sea = useConditions(curl);
  const v = useMemo(() => buildToday(inputs, { fishery, sea, place: { key: port.key, name: port.name } }), [inputs, fishery, sea, port.key, port.name]);
  const change = useChangeLine(port.key, sea !== null, v.allFacts, v.now);
  const savedAt = useOfflineCopy(port.key, port.name, curl, sea !== null, inputs.now);
  const [open, setOpen] = useState<Provenance | null>(null);
  const tour = useTourState();
  const phone = usePhone();
  const compact = useMedia(COMPACT);
  const [touring, setTouring] = useState(false);

  // the camera: the port and the water that matters, with room for the soundings
  const pts = v.soundings.map((d) => ({ lon: d.lon, lat: d.lat }));
  if (distanceKm(port, canyonHead) < 25) pts.push(canyonHead);
  const bounds = phone ? phoneBounds(port, pts) : todayBounds(port, pts, compact ? { minHalfKm: 14 } : {});
  const sceneId = `today:${port.key}`;
  const st = v.statement;
  // dated items the gates do not already hold: when the notice took effect, later assessments, coming NWS periods
  const ahead = agendaItems(v).filter((it) => it.id !== "today" && it.id !== "test" && !it.id.startsWith("alert-"));

  return (
    <>
      <ChartScene scene={{ id: sceneId, view: { bounds }, focus: { key: port.key, name: port.name, lon: port.lon, lat: port.lat } }} />
      <TodaySoundings items={v.soundings} sceneId={sceneId} onOpen={(id) => setOpen(v.soundings.find((d) => d.id === id)?.provenance ?? null)} />
      {open ? <ProvenanceSheet provenance={open} now={v.now} open onOpenChange={(o) => !o && setOpen(null)} /> : null}

      <div className={s.lead}>
        {st.state === "ok" ? (
          <div data-tour="status">
            <Statement
              size="md"
              as="h1"
              id="nx-today-title"
              tone={st.tone}
              rule={st.tone === "regulation" && !phone}
              scope={
                <span className={s.scopeInner}>
                  <motion.span layoutId={PORT_NAME_LAYOUT} className={s.portName} transition={pan}>
                    {port.name}
                  </motion.span>
                  <span className={s.scopeFishery}> · {scopeWords(st.fisheryLabel)}</span>
                </span>
              }
              title={st.title}
              why={st.why}
            >
              {st.provenance ? (
                <span data-tour="stamp">
                  <Stamp provenance={st.provenance} now={v.now} />
                </span>
              ) : null}
            </Statement>
            {st.conflict ? (
              <SourcesDiffer
                className={s.block}
                subject={st.fisheryLabel}
                claims={[
                  { agency: st.conflict.notice.agency, quote: st.conflict.notice.title, reference: "Official notices registry", url: st.conflict.notice.url ?? "https://wildlife.ca.gov" },
                  { agency: st.conflict.record.source, quote: st.conflict.record.headline, reference: "Neritic's season list", url: st.conflict.record.url },
                ]}
                checked={st.conflict.record.checked}
                now={v.now}
                next={<>Check {st.conflict.shown === "notice" ? st.conflict.notice.agency : st.conflict.record.source} before you act.</>}
              />
            ) : null}
          </div>
        ) : (
          <div data-tour="status">
            <p className={s.scope}>
              <motion.span layoutId={PORT_NAME_LAYOUT} className={s.portName} transition={pan}>
                {port.name}
              </motion.span>
              <span className={s.scopeFishery}> · {scopeWords(st.fisheryLabel)}</span>
            </p>
            <h1 id="nx-today-title" className={s.vh}>
              Today at {port.name}
            </h1>
            <Unavailable
              what="Season status isn't available"
              why="Neritic's season list did not load. Not knowing the status is not the same as open."
              next={
                <Button variant="text" href="https://wildlife.ca.gov/Fishing/Ocean/Regulations" external>
                  CDFW ocean regulations
                </Button>
              }
            />
          </div>
        )}

        {/* NWS advisories in force: under the status, in the first screen at every size (lead decision 6) */}
        <Advisories label="In force now" items={v.advisories.map((a) => ({ id: a.id, text: a.text, source: a.provenance ? <Stamp provenance={a.provenance} now={v.now} /> : "NWS" }))} />

        {/* what changed since the last visit sits with the status it may have changed */}
        <Change line={change} />

        {/* the next date that can change the status: said large, once (the whale gate below does not repeat it) */}
        {v.decision ? (
          <motion.div className={s.decision} initial={false} animate={{ opacity: 1 }} transition={settle}>
            <BigDate label={v.decision.label} start={v.decision.start} end={v.decision.end} now={v.now} size="lg">
              {v.decision.provenance && !v.decision.sameAs ? <Stamp provenance={v.decision.provenance} now={v.now} /> : null}
            </BigDate>
          </motion.div>
        ) : st.state === "ok" ? (
          <p className={s.quietLine}>No decision date on record for this fishery.</p>
        ) : null}

        <div className={s.actions}>
          {st.state === "ok" && st.source ? (
            <Button variant="primary" href={st.source.url} external>
              Read {st.source.agency}&rsquo;s notice
            </Button>
          ) : (
            <Button variant="primary" href={gateHrefs.trip(port.key)}>
              Plan a trip
            </Button>
          )}
          <span className={s.quiet}>
            <Button variant="text" href={gateHrefs.alerts(port.key)}>
              Alerts for {port.name}
            </Button>
          </span>
        </div>
        {/* first-use guidance for alerts: one dismissible line, then a quiet "Following" line (src/nx/alerts) */}
        <div data-slot="follow-line">
          <FollowLine port={{ key: port.key, name: port.name }} />
        </div>
        {tour === "" ? <TourLink onStart={() => setTouring(true)} /> : null}
      </div>

      <motion.div className={s.gatesWrap} key={`${port.key}:${fishery}`} initial={registerFrom} animate={registerTo} transition={register} data-tour="gates">
        <h2 className={s.gatesHead}>What decides a trip</h2>
        <GateList label={`What decides a trip from ${port.name}`} level={3} gates={v.gates.map((g) => gateItem(g, v.now, !!v.decision))} />
      </motion.div>

      {ahead.length ? (
        <div className={s.agendaWrap} data-tour="agenda">
          <h2 className={s.head}>Dates on record</h2>
          <Agenda className={s.agenda} label={`Dates at ${port.name}`} now={v.now} items={ahead} />
        </div>
      ) : null}

      <MarginRows className={s.rows} label={`More at ${port.name}`}>
        {v.rows
          // the gates hold the whale measures, the tests and the sea; what is left is the models
          .filter((r) => r.id === "models")
          .map((r) => (r.state === "unavailable" ? <MarginRow key={r.id} label={r.label} state="unavailable" note={r.note} /> : <MarginRow key={r.id} label="Models" value={<Figure text={r.value} />} detail={<Models v={v} />} />))}
        <NoticesRow notices={inputs.notices} portKey={port.key} portName={port.name} />
        {v.others.length ? (
          <MarginRow
            label="Others"
            value={<Figure text={othersLine(v.others)} />}
            detail={
              <ul className={s.detailList}>
                {v.others.map((o) => (
                  <li key={o.fishery}>
                    <p>
                      <strong>{o.label}</strong>: {o.word.toLowerCase()}. {o.headline}
                    </p>
                    {o.provenance ? <Stamp provenance={o.provenance} now={v.now} /> : null}
                    <p>
                      <Button variant="text" onClick={() => chooseFishery(o.fishery)}>
                        Lead with {o.label.replace(/\s*\(.*\)$/, "").toLowerCase()}
                      </Button>
                    </p>
                  </li>
                ))}
                <li>
                  <p>
                    <Button variant="text" href={noticesHref(port.key)}>
                      Every official notice, here and on the rest of the coast
                    </Button>
                  </p>
                </li>
              </ul>
            }
          />
        ) : null}
      </MarginRows>

      <Legend className={s.legend} kinds={["measured", "model", "official", "noview", "stale"]} title="Marks on this page" />
      <p className={s.imprint}>
        {v.dataAt ? <>Records published {dayTime(v.dataAt, { ref: v.now })}. </> : null}
        {savedAt ? <>Saved on this device for offline use {dayTime(savedAt, { ref: v.now })}. </> : null}
        Not for navigation.
      </p>
      {touring ? <Tour hasSounding={v.soundings.length > 0} onEnd={() => setTouring(false)} /> : null}
    </>
  );
}

/**
 * A gate of the view model as the ds GateList draws it: every line of fact carries its stamp (which opens where
 * it comes from), a sample is read with its age (T6, T17), the fixed relationship sentences read as one
 * paragraph. The sea gate's headline is its first fact (an advisory when one is in force). The whale gate's
 * next assessment is not repeated when the page already sets that date large.
 */
function gateItem(g: Gate, now: string, dateShown: boolean): GateItem {
  const lead = g.id === "sea" && g.state === "ok" ? g.facts[0] : null;
  const facts = (lead ? g.facts.slice(1) : g.facts).filter((f) => !(dateShown && g.id === "whale" && f.id === "next"));
  return {
    id: g.id,
    title: g.title,
    authority: g.authority,
    state: g.state,
    headline: (
      <span data-role={lead?.role} data-testid={`nx-gate-${g.id}-headline`}>
        {g.headline}
        {g.dateRole === "sampled" && g.age ? <span className={s.age}> · {g.age}</span> : null}
      </span>
    ),
    date: g.provenance ? <Stamp provenance={g.provenance} now={now} /> : undefined,
    facts: facts.map((f) => ({ id: f.id, text: f.text, role: f.role, old: f.old, stamp: f.provenance ? <Stamp provenance={f.provenance} now={now} /> : undefined })),
    relation: g.relation.join(" "),
    actions: g.actions,
  };
}

const noticesHref = (key: string) => `/notices?place=${encodeURIComponent(key)}`;

/**
 * The notices row: how many official notices the registry ties to this port, their titles, and the way to the
 * full list (every notice here and on the rest of the coast, each with its dates, sources and "not verified").
 */
function NoticesRow({ notices, portKey, portName }: { notices: TodayInputs["notices"]; portKey: string; portName: string }) {
  if (notices === null) return <MarginRow label="Notices" state="unavailable" note="The official notices did not load. Check CDFW and CDPH." />;
  const shown = notices.filter((n) => n.action !== "reopening" && n.action !== "advisory_lifted");
  return (
    <MarginRow
      label="Notices"
      value={<Figure text={shown.length ? `${shown.length} official, not verified` : "None listed here"} />}
      detail={
        <ul className={s.detailList} data-testid="nx-notices-row">
          {shown.map((n) => (
            <li key={n.id}>
              <p>
                <strong>{n.agency}</strong>: {n.title}
              </p>
            </li>
          ))}
          {!shown.length ? <li><p>The registry ties no current notice to {portName}. That does not mean open: check CDFW and CDPH.</p></li> : null}
          <li>
            <p>
              <Button variant="text" href={noticesHref(portKey)} data-testid="nx-notices-link">
                All notices, with dates and sources
              </Button>
            </p>
          </li>
        </ul>
      }
    />
  );
}

/** A value that changes settles in (README §8 `settle`); the first one is simply there. */
function Figure({ text }: { text: string }) {
  return (
    <AnimatePresence mode="popLayout" initial={false}>
      <motion.span key={text} className={s.figure} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={settle}>
        {text}
      </motion.span>
    </AnimatePresence>
  );
}

function agendaItems(v: View): AgendaItem[] {
  const statementStamp = v.statement.state === "ok" ? v.statement.provenance : null;
  return v.agenda.map((a) => ({
    id: a.id,
    start: a.start,
    end: a.end ?? null,
    title: a.title,
    detail: a.detail ?? undefined,
    tone: a.tone,
    tentative: a.tentative,
    // the legend's mark for the item's kind (T2): an NWS forecast is a prediction, never a solid "measured" dot
    kind: a.provenance ? markFor(a.provenance.kind) : undefined,
    // a stamp already shown above (the status's, the decision's) is not repeated
    stamp: a.provenance && a.provenance !== statementStamp && a.provenance !== v.decision?.provenance ? <Stamp provenance={a.provenance} now={v.now} /> : undefined,
  }));
}

function Change({ line }: { line: ChangeLine | null | undefined }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  if (!line) return null;
  if (!line.changes.length)
    return (
      <p className={s.change} data-testid="nx-change">
        {line.text}
      </p>
    );
  return (
    <div className={s.changeWrap}>
      <button type="button" className={s.changeToggle} aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)} data-testid="nx-change">
        {line.text}
        <span className={s.more}>{open ? "Less" : "What changed"}</span>
      </button>
      <AnimatePresence initial={false}>
        {open ? (
          <motion.ul id={id} className={s.changeList} initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={settle}>
            {line.changes.map((c) => (
              <li key={c.id}>
                <strong>{c.change === "new" ? "New" : c.change === "gone" ? "No longer listed" : "Changed"}.</strong> {(c.after ?? c.before)!.headline}
                {c.change === "changed" && c.before && c.before.headline !== c.after?.headline ? <span className={s.before}> Before: {c.before.headline}</span> : null}
              </li>
            ))}
          </motion.ul>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

function Models({ v }: { v: View }) {
  const m = v.models;
  return (
    <div className={s.models}>
      {m.charm ? (
        <div>
          <h3 className={s.subhead}>C-HARM{m.charm.past ? ", past run" : ""}</h3>
          <p>{m.charm.headline}</p>
          <p>{m.charm.words}</p>
          {m.charm.provenance ? <Stamp provenance={m.charm.provenance} now={v.now} /> : null}
        </div>
      ) : (
        <p>No C-HARM run is available for these days.</p>
      )}
      {m.outlook ? (
        <div>
          <h3 className={s.subhead}>Chlorophyll outlook, 8 days</h3>
          <p>{m.outlook.headline}</p>
          {m.outlook.words.map((w) => (
            <p key={w}>{w}</p>
          ))}
          {m.outlook.provenance ? <Stamp provenance={m.outlook.provenance} now={v.now} /> : null}
        </div>
      ) : (
        <p>No chlorophyll outlook is published for this port.</p>
      )}
    </div>
  );
}
