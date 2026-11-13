"use client";

import { AnimatePresence, motion, MotionConfig } from "motion/react";
import { useEffect, useId, useMemo, useState } from "react";
import type { FisheryId } from "@/pilot/contracts";
import { fisheryLabel as labelOf, nextTide, pacificDay, rowHeadline } from "@/pilot/facts";
import { readSaved } from "@/pilot/today/prefs";
import { saveTodayOffline } from "@/pilot/today/offline";
import { useFollow } from "@/nx/alerts/useFollow";
import { ChartRegulation, RAMP_FISHERIES, RAMP_ZONES, rampZoneAt } from "@/nx/chart/regulation";
import { ChartScene } from "@/nx/chart/scene";
import { setMapLayer, MapLayers } from "@/nx/chart/mapLayers";
import { Agenda, type AgendaItem } from "@/nx/ds/Agenda";
import { Button } from "@/nx/ds/Button";
import { dayTime } from "@/nx/ds/format";
import { markFor } from "@/nx/ds/kinds";
import { Legend } from "@/nx/ds/Mark";
import { PageLayout } from "@/nx/ds/layout";
import { settle } from "@/nx/ds/motion";
import { ProvenanceSheet, Stamp, stampText, type Provenance } from "@/nx/ds/Stamp";
import { SourcesDiffer, Unavailable } from "@/nx/ds/States";
import { Skeleton } from "@/nx/ds/DataState";
import { Statement } from "@/nx/ds/Statement";
import { Ticket, TicketMain, TicketStub } from "@/nx/ds/stage/Ticket";
import { FollowButton } from "@/nx/ds/synth/FollowButton";
import { Glyph, LimitGauge, Micro, Pennant, Slot } from "@/nx/ds/synth/Grammar";
import { Tabs, TabPanel } from "@/nx/ds/synth/Tabs";
import { useMedia, usePhone } from "@/nx/ds/useMedia";
import { CANYON_AXIS } from "@/nx/chart/features";
import { sectorOf, whaleMeasures } from "@/nx/regulation/whale";
import { distanceKm, portWater, zoneBounds } from "./camera";
import type { ChangeLine } from "./change";
import { useChangeLine } from "./useChangeLine";
import { conditionsUrl, useConditions, useTodayInputs, type Initial } from "./data";
import { FisheryChooser } from "./FisheryChooser";
import { GateList, type GateItem } from "@/nx/ds/GateList";
import { gateHrefs, type Gate } from "./gates";
import { buildToday, ppmWords, scopeWords, type TodayInputs, type TodayView as View } from "./model";
import { advisoryParts, SeaDock } from "./SeaDock";
import { TodaySoundings } from "./Soundings";
import { chooseFishery, useFisheries, useTourState } from "./store";
import { compassDeg, datesLine, glyphFor, pennantsFor, posterParts, seaBars, zoneCaption } from "./ticket";
import { DatesLine } from "@/nx/ds/synth/charts";
import { Tour, TourLink } from "./Tour";
import s from "./today.module.css";

export type TodayProps = {
  port: { key: string; name: string; lon: number; lat: number } | null;
  initial: Initial;
  /** first visit at `/`: the chart stays on the coast until a fishery is chosen */
  arriving?: boolean;
};

const canyonHead = { lon: CANYON_AXIS[0][0], lat: CANYON_AXIS[0][1] };
const TICKET_W = 404;

/**
 * Today at your port (v14, the approved synthesis "Today"): the chart is the page. One ticket stands beside the port
 * and says one thing: the season status as a poster ("Season delayed to at least **Jan 1, 2027**", the record's own
 * words), its stamp, and four tabs: Status (why, the dates on record, the domoic-acid test), Record (the source record
 * in detail: the three gates, the dates, when the records were published and saved), Fisheries (every fishery at the
 * port) and Notices (official notices, then the models, quietly). The sea dock along the foot holds the NWS zone,
 * advisories, the forecast seas, the buoy's point and the tide. Readings live on the chart at their points, under their
 * layer chips. Every value comes from the published records and opens where it comes from.
 */
export function TodayView({ port, initial, arriving = false }: TodayProps) {
  const fisheries = useFisheries();
  const data = useTodayInputs(port?.key ?? null, initial);
  const primary = fisheries?.[0];
  if (!port) return null;
  const inputs = data.state === "ok" ? data.inputs : null;
  const ramp = !!fisheries?.some((f) => RAMP_FISHERIES.has(f));
  const assembled = fisheries !== undefined && !!primary && data.state === "ok" && !!inputs;
  const at = { lon: port.lon, lat: port.lat };
  const scene = <ChartScene scene={{ id: `today:${port.key}`, view: { bounds: portWater(port) }, focus: { key: port.key, name: port.name, lon: port.lon, lat: port.lat } }} />;
  return (
    <MotionConfig reducedMotion="user">
      <PageLayout mode="stage" phoneMap={0.27} />
      <MapLayers page="today" defaults={{ regulation: true, depth: true, sea: true }} />
      {!assembled ? <ChartRegulation spec={ramp ? { zones: { highlight: rampZoneAt(port.lat) } } : null} /> : null}
      <section className={s.today} data-testid="nx-today" aria-labelledby="nx-today-title">
        {fisheries === undefined ? (
          <Ticket key={port.key} at={at} anchorY={110} width={TICKET_W} labelledBy="nx-today-title">
            <TicketMain>
              <PortHeading name={port.name} />
            </TicketMain>
          </Ticket>
        ) : !primary ? (
          <>
            {!arriving ? scene : null}
            <Ticket key={port.key} at={arriving ? null : at} anchorY={110} width={TICKET_W} labelledBy="nx-today-title">
              <TicketMain>
                <FisheryChooser port={port} inputs={data.state === "loading" ? undefined : inputs} onChoose={chooseFishery} />
              </TicketMain>
            </Ticket>
          </>
        ) : data.state === "loading" ? (
          <>
            {scene}
            <Ticket key={port.key} at={at} anchorY={170} width={TICKET_W} labelledBy="nx-today-title">
              <TicketMain>
                <PortHeading name={port.name} fishery={primary} />
                {/* a placeholder at the ticket's own geometry: the poster, then the tabs' first lines */}
                <Skeleton label={`Reading the records for ${port.name}`} figure lines={4} />
              </TicketMain>
            </Ticket>
          </>
        ) : !assembled ? (
          <>
            {scene}
            <Ticket key={port.key} at={at} anchorY={170} width={TICKET_W} labelledBy="nx-today-title">
              <TicketMain>
                <PortHeading name={port.name} fishery={primary} />
                <Unavailable
                  what={`Today at ${port.name} could not be read`}
                  why="Neritic's published records did not load. The agencies' own pages still work."
                  next={
                    <p className={s.links}>
                      <Button variant="quiet" href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" external>
                        CDFW advisories
                      </Button>
                      <Button variant="quiet" href="https://www.weather.gov/mtr/" external>
                        NWS marine forecast
                      </Button>
                    </p>
                  }
                />
              </TicketMain>
            </Ticket>
          </>
        ) : (
          <Assembled port={port} inputs={inputs!} fishery={primary!} bands={data.state === "ok" ? data.bands : {}} />
        )}
      </section>
    </MotionConfig>
  );
}

function PortHeading({ name, fishery }: { name: string; fishery?: FisheryId }) {
  return (
    <>
      <Micro as="p">
        {name}
        {fishery ? ` · ${scopeWords(labelOf(fishery))}` : ""}
      </Micro>
      <h1 id="nx-today-title" className={s.vh}>
        Today at {name}
      </h1>
    </>
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

type Tab = "status" | "record" | "fisheries" | "notices";
const noticesHref = (key: string) => `/notices?place=${encodeURIComponent(key)}`;
const CDFW_ADVISORIES = "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories";

function Assembled({ port, inputs, fishery, bands }: { port: NonNullable<TodayProps["port"]>; inputs: TodayInputs; fishery: FisheryId; bands: Record<string, { south: number; north: number }> }) {
  const curl = conditionsUrl(inputs.place.lat, inputs.place.lon);
  const sea = useConditions(curl);
  const v = useMemo(() => buildToday(inputs, { fishery, sea, place: { key: port.key, name: port.name } }), [inputs, fishery, sea, port.key, port.name]);
  const change = useChangeLine(port.key, sea !== null, v.allFacts, v.now);
  const savedAt = useOfflineCopy(port.key, port.name, curl, sea !== null, inputs.now);
  const [open, setOpen] = useState<Provenance | null>(null);
  const [tab, setTab] = useState<Tab>("status");
  const [area, setArea] = useState<string | null>(null);
  const tour = useTourState();
  const phone = usePhone();
  const [touring, setTouring] = useState(false);
  // one follow state with Alerts and the chart's port mark (src/nx/alerts/useFollow)
  const fol = useFollow(port.key);
  const following = !!fol.following;
  const tabsId = `nx-today-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;

  // the camera: the port and the water that matters, with room for the readings
  const pts = v.soundings.map((d) => ({ lon: d.lon, lat: d.lat }));
  if (distanceKm(port, canyonHead) < 25) pts.push(canyonHead);
  const ramp = RAMP_FISHERIES.has(fishery);
  const zoneId = ramp ? rampZoneAt(port.lat) : null;
  const bounds = zoneBounds(port, RAMP_ZONES.find((z) => z.zone === zoneId) ?? null, pts);
  const sceneId = `today:${port.key}`;
  const st = v.statement;
  const year = Number(pacificDay(v.now).slice(0, 4));
  const poster = st.state === "ok" ? posterParts(st.title, year) : null;

  // the whale zone on the water, captioned in the record's words (RAMP readers only)
  const whale = v.model.whale;
  const whaleDelay = st.state === "ok" && st.row.cause === "whale_entanglement";
  const caption = ramp && whale ? zoneCaption(whaleDelay ? st.level : null, whaleMeasures(whale, sectorOf(fishery), { seasonSaysIt: whaleDelay }).short.split(", ")[0]) : null;
  // an advisory area a Fisheries row put on the chart (rock crab's "do not eat internal organs", for one)
  const notices = (inputs.notices ?? []).filter((n) => n.action !== "reopening" && n.action !== "advisory_lifted");
  const areaNotice = area ? notices.find((n) => n.id === area) : null;
  // hatched on the water from shore to the frame's edge, in the record's words, with its "not verified" stamp
  const advisory = areaNotice && bands[areaNotice.id] ? { id: areaNotice.id, north: bands[areaNotice.id].north, south: bands[areaNotice.id].south, title: areaNotice.title, note: `${areaNotice.agency} · official, not verified` } : null;

  // the dates on record: the delay from its effective date to the earliest start, today, the next assessment
  const eff = v.agenda.find((a) => a.id === "notice")?.start ?? null;
  const dl =
    st.state === "ok"
      ? datesLine({
          today: pacificDay(v.now),
          effective: eff,
          earliest: st.level === "delayed" ? (poster?.date ?? null) : null,
          window: v.decision ? { start: v.decision.start, end: v.decision.end } : null,
          windowWords: v.decision?.label.replace(/, subject to change$/, "").toLowerCase(),
        })
      : null;

  // the sea: NWS's periods, its advisories, the buoy's one point
  const seaOk = sea && sea !== "error" ? sea : null;
  const bars = seaOk
    ? seaBars({
        lanes: v.model.sea.lanes,
        now: v.now,
        alerts: seaOk.alerts.status === "ok" ? seaOk.alerts.data.filter((a) => !a.ends || Date.parse(a.ends) > Date.parse(v.now)) : [],
        buoy: v.model.sea.buoy,
      })
    : null;
  const tideNext = v.model.sea.state === "ok" ? nextTide(v.model.sea.tides, v.now) : null;
  const tide = tideNext ? `${tideNext.type === "H" ? "high" : "low"} ${tideNext.ft < 0 ? "−" : ""}${Math.abs(tideNext.ft).toFixed(1)} ft · ${clock(tideNext.at)}` : null;
  const b = v.model.sea.buoy;
  const windKt = b?.windKt ?? v.model.sea.windBuoy?.windKt ?? null;
  const windDir = b?.windKt != null ? b.windDir : (v.model.sea.windBuoy?.windDir ?? null);
  const water = v.model.sea.water;
  const extra = {
    windDeg: compassDeg(windDir),
    wind: [windKt != null ? `${windDir ? `${windDir} ` : ""}${Math.round(windKt)} kt` : null, water ? `${Math.round((water.c * 9) / 5 + 32)} °F` : null].filter(Boolean).join(" · ") || null,
  };

  // the fisheries at this port: the one Today leads with, then the others, each with its stamp
  const fisheries = [
    ...(st.state === "ok" ? [{ fishery, label: st.fisheryLabel, word: st.row.word, headline: rowHeadline(st.row), flagged: st.tone !== "ink", provenance: st.provenance, lead: true }] : []),
    ...v.others.map((o) => ({ ...o, lead: false })),
  ];
  const areaRows = notices.filter((n) => bands[n.id] && n.action === "consumption_advisory");

  const tabs = [
    { id: "status" as const, label: "Status" },
    { id: "record" as const, label: "Record" },
    { id: "fisheries" as const, label: "Fisheries", count: fisheries.length },
    { id: "notices" as const, label: "Notices", count: inputs.notices === null ? null : notices.length },
  ];
  const cdfw = st.state === "ok" && st.source ? st.source : { agency: "CDFW", url: CDFW_ADVISORIES };

  return (
    <>
      <ChartScene scene={{ id: sceneId, view: { bounds }, focus: { key: port.key, name: port.name, lon: port.lon, lat: port.lat } }} />
      <ChartRegulation spec={{ zones: ramp ? { highlight: zoneId, caption } : null, advisory }} />
      <TodaySoundings items={v.soundings} sceneId={sceneId} extra={extra} onOpen={(id) => setOpen(v.soundings.find((d) => d.id === id)?.provenance ?? null)} />
      {open ? <ProvenanceSheet provenance={open} now={v.now} open onOpenChange={(o) => !o && setOpen(null)} /> : null}

      <Ticket key={port.key} at={{ lon: port.lon, lat: port.lat }} anchorY={170} top={208} width={TICKET_W} labelledBy="nx-today-title" className={s.ticket}>
        <TicketMain data-tour="status">
          <Micro as="p" className={s.scope}>
            {port.name} · {scopeWords(st.fisheryLabel)}
          </Micro>
          {st.state === "ok" ? (
            <>
              {poster ? (
                <Statement as="h1" id="nx-today-title" tone={st.tone} lede={poster.lede} hero={poster.hero} className={s.poster} />
              ) : (
                <Statement as="h1" id="nx-today-title" size="md" tone={st.tone} title={st.title} className={s.poster} />
              )}
              {st.provenance ? (
                <span data-tour="stamp" className={s.stampWrap}>
                  <TicketStamp provenance={st.provenance} now={v.now} />
                </span>
              ) : null}
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
            </>
          ) : (
            <>
              <h1 id="nx-today-title" className={s.vh}>
                Today at {port.name}
              </h1>
              <Unavailable
                what="Season status isn't available"
                why="Neritic's season list did not load. Not knowing the status is not the same as open."
                next={
                  <Button variant="quiet" href="https://wildlife.ca.gov/Fishing/Ocean/Regulations" external>
                    CDFW ocean regulations
                  </Button>
                }
              />
            </>
          )}
          {/* what changed since the last visit sits with the status it may have changed */}
          <Change line={change} />
          {/* a phone: the advisories in force above the fold, in the ticket (the dock follows below) */}
          {phone && v.advisories.length ? (
            <div className={s.advStrip} data-testid="advisories">
              {v.advisories.map((a) => {
                const p = advisoryParts(a);
                return (
                  <p key={a.id}>
                    <Pennant n={pennantsFor(p.event)} />
                    <span>
                      <b>{p.event}</b> {p.when}
                    </span>
                  </p>
                );
              })}
              {v.advisories[0].provenance ? <Stamp provenance={v.advisories[0].provenance} now={v.now} className={s.advStamp} /> : null}
            </div>
          ) : null}
        </TicketMain>
        <TicketStub data-tour="gates">
          <Tabs label="Detail" items={tabs} value={tab} onChange={setTab} idBase={tabsId} className={s.tabs} />
          <TabPanel idBase={tabsId} id="status" active={tab === "status"} className={s.panel}>
            <StatusPanel v={v} dl={dl} researchHref={gateHrefs.research(port.key)} />
          </TabPanel>
          <TabPanel idBase={tabsId} id="record" active={tab === "record"} className={s.panel}>
            <RecordPanel v={v} portName={port.name} savedAt={savedAt} />
          </TabPanel>
          <TabPanel idBase={tabsId} id="fisheries" active={tab === "fisheries"} className={s.panel}>
            <Micro as="p" className={s.panelHead}>
              Fisheries at {port.name} · official, not verified
            </Micro>
            <ul className={s.fl}>
              {fisheries.map((f) => (
                <li key={f.fishery} data-fishery={f.fishery}>
                  <Glyph kind={f.provenance ? glyphFor(f.provenance, f.flagged) : "none"} />
                  <span>
                    {f.label}
                    <small className={f.flagged ? s.coral : undefined}>{f.word}</small>
                    {!f.lead ? (
                      <Button variant="quiet" onClick={() => chooseFishery(f.fishery)} className={s.lead}>
                        Lead with {f.label.replace(/\s*\(.*\)$/, "").toLowerCase()}
                      </Button>
                    ) : null}
                  </span>
                </li>
              ))}
            </ul>
            {areaRows.length ? (
              <ul className={s.areas}>
                {areaRows.map((n) => (
                  <li key={n.id}>
                    {/* design-lint-allow raw-button: a toggle row that puts an advisory's area on the chart (prototype .fish-pick); ds Toggle is a switch, this is the row itself */}
                    <button
                      type="button"
                      className={s.pick}
                      aria-pressed={area === n.id}
                      onClick={() => {
                        const on = area !== n.id;
                        setArea(on ? n.id : null);
                        if (on) setMapLayer("regulation", true);
                      }}
                    >
                      <Glyph kind="unverified-coral" />
                      <span>
                        {n.title}
                        <small>
                          {n.agency} · not verified · <u>{area === n.id ? "on the chart" : "show on the chart"}</u>
                        </small>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
          </TabPanel>
          <TabPanel idBase={tabsId} id="notices" active={tab === "notices"} className={s.panel}>
            <NoticesPanel notices={inputs.notices === null ? null : notices} portKey={port.key} portName={port.name} v={v} />
          </TabPanel>
          <div className={s.acts}>
            {following ? (
              <>
                {/* design issue 1: a followed port's ticket leads with the next useful action; Following is a quiet toggle */}
                <Button variant="primary" href={gateHrefs.trip(port.key)}>
                  Plan a trip
                </Button>
                <FollowButton name={port.name} following variant="toggle" onChange={fol.set} data-testid="nx-follow" />
              </>
            ) : (
              <FollowButton name={port.name} following={false} disabled={fol.following === undefined || fol.full} onChange={fol.set} data-testid="nx-follow" />
            )}
            <Button variant="quiet" href={cdfw.url} external>
              {cdfw.agency}
            </Button>
          </div>
          {tour === "" ? <TourLink onStart={() => setTouring(true)} /> : null}
        </TicketStub>
      </Ticket>

      <SeaDock state={v.sea} zone={v.model.sea.zone} zoneName={seaOk?.marine.status === "ok" ? seaOk.marine.data.zoneName : null} advisories={v.advisories} bars={bars} tide={tide} now={v.now} phone={phone} tripHref={following ? null : gateHrefs.trip(port.key)} />
      {touring ? <Tour hasSounding={v.soundings.length > 0} onEnd={() => setTouring(false)} /> : null}
    </>
  );
}

const clock = (at: string) => {
  const m = /(\d{2}):(\d{2})/.exec(at);
  if (!m) return at;
  const h = Number(m[1]);
  return `${h % 12 || 12}:${m[2]} ${h < 12 ? "AM" : "PM"}`;
};

/** The ticket's stamp: the data-state glyph, then source · date · verification; it opens where the record comes from. */
function TicketStamp({ provenance: p, now }: { provenance: Provenance; now: string }) {
  const [open, setOpen] = useState(false);
  const text = stampText(p, now).replace(/^[^·]+·\s*/, "");
  return (
    <ProvenanceSheet
      provenance={p}
      now={now}
      open={open}
      onOpenChange={setOpen}
      trigger={
        // design-lint-allow raw-button: the stamp line is the provenance sheet's trigger (ds Stamp, in the synthesis grammar)
        <button type="button" className={s.stamp} aria-haspopup="dialog" aria-label={`${stampText(p, now)}. Open where this comes from`}>
          <Glyph kind={glyphFor(p)} />
          <span>{text}</span>
        </button>
      }
    />
  );
}

/** Status: why, the dates on record (drawn), the domoic-acid test or the honest slot. */
function StatusPanel({ v, dl, researchHref }: { v: View; dl: ReturnType<typeof datesLine>; researchHref: string }) {
  const st = v.statement;
  // a short screen (prototype @media max-height 800): the dates line is drawn lower so the ticket's action stays in view
  const short = useMedia("(max-height: 800px) and (min-width: 641px)");
  const t = v.test;
  return (
    <>
      {st.state === "ok" && st.why ? (
        <p className={s.why}>
          <Micro>Why</Micro>
          <b>{st.why}</b>
        </p>
      ) : null}
      {v.decision ? (
        <div className={s.tlFig}>
          <p className={s.tlCap} data-testid="nx-decision">
            <Glyph kind="predicted" />
            <span>
              {v.decision.label} · <b>{rangeWords(v.decision.start, v.decision.end)}</b>
            </span>
          </p>
          {dl ? <DatesLine {...dl} height={short ? 66 : 82} /> : null}
        </div>
      ) : st.state === "ok" ? (
        <p className={s.quietLine}>No decision date on record for this fishery.</p>
      ) : null}
      <div className={s.cell} data-state={t.state === "ok" ? (t.old ? "old" : "ok") : t.state} data-testid="nx-test-cell">
        <Micro>Domoic acid test</Micro>
        {t.state === "none" ? (
          <Slot>No test site for this port</Slot>
        ) : t.state === "error" ? (
          <Slot>CDPH tests could not be read just now</Slot>
        ) : (
          <div className={s.tox}>
            <span className={s.toxHead}>
              <Glyph kind={t.old ? "old" : "measured"} />
              {t.species}, highest {t.max == null ? "not detected" : `${ppmWords(t.max)}${t.limit != null ? ` of ${t.limit}` : ""} ppm`}
            </span>
            {t.max != null && t.limit != null ? <LimitGauge value={t.max} limit={t.limit} label={`${ppmWords(t.max)} against an action level of ${t.limit} ppm`} /> : null}
            <small>
              {t.n} sample{t.n === 1 ? "" : "s"} · {t.site} · {t.day} · {t.age}
              {t.old ? " (old)" : ""}
            </small>
          </div>
        )}
        {/* the water samples near the port, in Research (the test's own record is CDPH's, linked from Record) */}
        <Button variant="quiet" href={researchHref} className={s.cellLink}>
          Samples nearby
        </Button>
      </div>
    </>
  );
}

const rangeWords = (a: string, b: string | null) => {
  const f = (d: string) => new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" }).format(new Date(`${d}T12:00:00Z`));
  if (!b || b === a) return f(a);
  return a.slice(0, 7) === b.slice(0, 7) ? `${f(a)}–${Number(b.slice(8))}` : `${f(a)}–${f(b)}`;
};

/** Record: the source record and the three gates in detail, the dates on record, when the records were published and saved. */
function RecordPanel({ v, portName, savedAt }: { v: View; portName: string; savedAt: string | null }) {
  const st = v.statement;
  const ahead = agendaItems(v).filter((it) => it.id !== "today" && it.id !== "test" && !it.id.startsWith("alert-"));
  return (
    <>
      {st.state === "ok" ? <p className={s.quote}>{rowHeadline(st.row)}</p> : null}
      <GateList label={`What decides a trip from ${portName}`} level={3} gates={v.gates.map((g) => gateItem(g, v.now, false))} />
      {ahead.length ? (
        <div className={s.agendaWrap} data-tour="agenda">
          <Micro as="h3" className={s.panelHead}>
            Dates on record
          </Micro>
          <Agenda label={`Dates at ${portName}`} now={v.now} items={ahead} />
        </div>
      ) : null}
      <Legend className={s.legend} kinds={["measured", "model", "official", "noview", "stale"]} title="Marks on this page" />
      <p className={s.fresh}>
        {v.dataAt ? <>Records published {dayTime(v.dataAt, { ref: v.now })}. </> : null}
        {savedAt ? <>Saved on this device for offline use {dayTime(savedAt, { ref: v.now })}. </> : null}
        Not for navigation.
      </p>
    </>
  );
}

/** Notices: the official notices the registry ties to this port (not verified), then the models, quietly: not toxin tests. */
function NoticesPanel({ notices, portKey, portName, v }: { notices: TodayInputs["notices"]; portKey: string; portName: string; v: View }) {
  const m = v.models;
  return (
    <>
      {notices === null ? (
        <p className={s.quietLine}>The official notices did not load. Check CDFW and CDPH.</p>
      ) : (
        <>
          <Micro as="p" className={s.panelHead}>
            {notices.length ? `${notices.length} official notice${notices.length === 1 ? "" : "s"} · not verified` : "Official notices"}
          </Micro>
          <ul className={s.nl} data-testid="nx-notices-row">
            {notices.map((n) => (
              <li key={n.id}>
                <Glyph kind="unverified" />
                <span>
                  <strong>{n.agency}</strong>: {n.title}
                </span>
              </li>
            ))}
            {!notices.length ? (
              <li>
                <Glyph kind="none" />
                <span>The registry ties no current notice to {portName}. That does not mean open: check CDFW and CDPH.</span>
              </li>
            ) : null}
          </ul>
        </>
      )}
      <p className={s.more}>
        <Button variant="quiet" href={noticesHref(portKey)} data-testid="nx-notices-link">
          All notices, with dates and sources
        </Button>
      </p>
      <Micro as="h3" className={s.panelHead}>
        Models · not toxin tests
      </Micro>
      <p className={s.modelsLine}>{m.value}</p>
      <ul className={s.nl} data-testid="nx-models">
        <li>
          <Glyph kind="predicted" />
          <span>
            {m.charm ? (
              <>
                <strong>C-HARM{m.charm.past ? ", past run" : ""}</strong>: {m.charm.headline} {m.charm.words}
                {m.charm.provenance ? (
                  <>
                    {" "}
                    <Stamp provenance={m.charm.provenance} now={v.now} />
                  </>
                ) : null}
              </>
            ) : (
              "No C-HARM run is available for these days."
            )}
          </span>
        </li>
        <li>
          <Glyph kind="predicted" />
          <span>
            {m.outlook ? (
              <>
                <strong>Chlorophyll outlook, 8 days</strong>: {m.outlook.headline} {m.outlook.words.join(" ")}
                {m.outlook.provenance ? (
                  <>
                    {" "}
                    <Stamp provenance={m.outlook.provenance} now={v.now} />
                  </>
                ) : null}
              </>
            ) : (
              "No chlorophyll outlook is published for this port."
            )}
          </span>
        </li>
      </ul>
    </>
  );
}

/**
 * A gate of the view model as the ds GateList draws it: every line of fact carries its stamp (which opens where
 * it comes from), a sample is read with its age (T6, T17), the fixed relationship sentences read as one
 * paragraph. The sea gate's headline is its first fact (an advisory when one is in force).
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
      <p className={s.changeLine} data-testid="nx-change">
        {line.text}
      </p>
      <Button variant="quiet" aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)} data-testid="nx-change-more">
        {open ? "Less" : "What changed"}
      </Button>
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
