"use client";

import { stableFormat } from "@/lib/time";
import "./today.css";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { PILOT_PORTS, type FisheryId, type Readiness } from "@/pilot/contracts";
import type { Conditions } from "@/lib/conditions";
import { DEFAULT_FISHERIES, FISHERY_OPTIONS, buildTodayModel, dayWords, rangeWords, type PortTodayInputs, type StatusRow, type TodayModel } from "@/pilot/facts";
import { OceanStage } from "@/v4/stage/OceanStage";
import type { StageData, StageMark } from "@/v4/stage/types";
import { useFocusLink } from "@/v4/motion";
import { Freshness } from "@/v4/ui/Freshness";
import { Unverified } from "@/v4/ui/Unverified";
import { Pop } from "@/v4/ui/Pop";
import { TopBar } from "@/v4/shell/TopBar";
import { useReportTrust, type Trust } from "@/v4/shell/Notices";
import { useReportSource } from "@/v4/readiness/store";
import { PilotDock, PilotFooter } from "@/pilot/shell/PilotShell";
import { SkipLink } from "@/v4/shell/SkipLink";
import { Icon } from "@/v4/ui/Icon";
import { readFisheries, writeFisheries, writeTodayPlace } from "./prefs";
import { savedMark } from "./offline";
import { FisheryPicker, StatusSheet, StatusStrip, WhaleSheet, statusFocusId } from "./Status";
import { SeaWeather } from "./Sea";
import { ToxinTests } from "./Tests";
import { BloomOutlook } from "./Bloom";
import { WhatChanged } from "./Changed";
import { Actions } from "./Actions";
import { FactAnchors, StatusMark, factAnchor, sevClass } from "./ui";
import { AlertCheck } from "@/pilot/alerts/AlertCheck";
import { AskSheet } from "@/pilot/ask/ui/AskSheet";
import { periodLine, windWords } from "./marine";

export type TodayProps = {
  /** what the top bar's trust pill says */
  trust: Trust | null;
  inputs: Omit<PortTodayInputs, "conditions" | "selected">;
  /** from the link (?f=…); null: the reader's saved list, else the default */
  linkFisheries: FisheryId[] | null;
  readiness: { alerts: Readiness; ask: Readiness; sms: Readiness };
  /** the server is reading the test fixture: alerts made here are marked as samples */
  fixture: boolean;
  /** what the ocean stage draws; null shows the poster */
  stage: StageData | null;
};

const ALL_FISHERIES = FISHERY_OPTIONS.map((o) => o.id);

/** The fact a link points at: `#fact-<PortFact.id>` (Ask Neritic's citations, alerts). */
function useFactTarget(): string | null {
  const [target, setTarget] = useState<string | null>(null);
  useEffect(() => {
    const read = () => {
      let h = window.location.hash.slice(1);
      try {
        h = decodeURIComponent(h);
      } catch {
        // a hash that is not valid percent-encoding: used as written
      }
      setTarget(h.startsWith("fact-") ? h.slice(5) : null);
    };
    read();
    window.addEventListener("hashchange", read);
    return () => window.removeEventListener("hashchange", read);
  }, []);
  return target;
}

const STAMP = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

/** Ocean conditions for the port from /api/conditions (NWS, NDBC, CO-OPS), with ten forecast periods (about five days). */
function useConditions(url: string): Conditions | null | "error" {
  const [c, setC] = useState<Conditions | null | "error">(null);
  useEffect(() => {
    let live = true;
    setC(null);
    fetch(url)
      .then((r) => (r.ok ? (r.json() as Promise<Conditions>) : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => live && setC(d))
      .catch(() => live && setC("error"));
    return () => {
      live = false;
    };
  }, [url]);
  return c;
}

type Bind = ReturnType<typeof useFocusLink>["bind"];
function Glance({ href, id, bind, label, children, sub, testid }: { href: string; id?: string | null; bind: Bind; label: ReactNode; children: ReactNode; sub: ReactNode; testid: string }) {
  return (
    <li>
      <a href={href} className="today-gl nx-focus nx-press" data-testid={testid} {...(id ? bind(id) : {})}>
        <span className="today-gl-k">{label}</span>
        <span className="today-gl-v">{children}</span>
        <span className="today-gl-s">{sub}</span>
      </a>
    </li>
  );
}

/** The glance row along the stage's foot: four items, unboxed, each a link to its chapter and each lighting its subject on the map. */
function GlanceRow({ model, now, bind }: { model: TodayModel; now: string; bind: Bind }) {
  const flagged = model.status.filter((r) => r.level !== "unknown" && r.level !== "no_restriction_on_record");
  const known = model.status.some((r) => r.level !== "unknown");
  const top = flagged[0] ? statusFocusId(flagged[0]) : null;
  const sea = model.sea;
  const wind = sea.buoy?.windKt != null ? { kt: sea.buoy.windKt, dir: sea.buoy.windDir } : sea.windBuoy ? { kt: sea.windBuoy.windKt, dir: sea.windBuoy.windDir } : null;
  const nowSea = [wind ? `${wind.dir ?? ""} ${Math.round(wind.kt)} kt`.trim() : null, sea.buoy?.waveFt != null ? `${sea.buoy.waveFt.toFixed(1)} ft` : null].filter(Boolean).join(", ");
  const next = sea.lanes[1];
  const nextWind = next ? windWords(next)?.split(", then ")[0] : null;
  const test = model.tests[0];
  const read = model.bloom.outlookRead;
  const outlook = model.bloom.outlook;
  return (
    <nav aria-label="This briefing" data-beat="3">
      <ul className="today-glance" data-testid="today-glance">
        <Glance bind={bind} href="#status" id={top} label={
            <>
              Fisheries
            </>
          } testid="glance-status" sub={known ? (flagged.length ? "not verified" : "not a promise they are open") : "status not available"}>
          {known ? (
            <>
              <span>{flagged.length ? `${flagged.length} flagged` : "None flagged"}</span>
              <span className="today-ribbon" aria-hidden>
                {model.status.map((r) => (
                  <StatusMark key={r.fishery} severity={r.severity} seasonal={r.seasonal} className="today-ribbon-m" />
                ))}
              </span>
            </>
          ) : (
            "Unknown"
          )}
        </Glance>
        <Glance bind={bind} href="#sea" id={sea.buoy ? `buoy:${sea.buoy.id}` : null} label="Sea" testid="glance-sea" sub={sea.state === "loading" ? "reading NWS and NDBC…" : next && nextWind ? `${next.name}: ${nextWind}` : "NWS, measured buoys"}>
          {nowSea || (sea.state === "loading" ? "…" : "No reading")}
        </Glance>
        <Glance bind={bind} href="#tests" id={test ? `site:${test.site.id}` : null} label="Tests" testid="glance-tests" sub={test?.latest ? `${dayWords(test.latest.date, now).replace(/, \d{4}$/, "")}${test.stale ? " (old)" : ""}` : model.testsError ? "not available" : "none published here"}>
          {test?.latest?.avg_ppm != null && test.action != null ? `${test.latest.avg_ppm} of ${test.action} ppm` : test?.latest ? "Not detected" : "No test"}
        </Glance>
        <Glance bind={bind} href="#bloom" label={outlook && !model.bloom.outlookStale ? <>Bloom<span className="today-gl-more">, {rangeWords(outlook.valid_from, outlook.valid_to, now)}</span></> : "Bloom"} testid="glance-bloom" sub="chlorophyll, not toxin">
          {read ? read.phrase.charAt(0).toUpperCase() + read.phrase.slice(1) : outlook ? "No usual to compare" : "No outlook"}
        </Glance>
      </ul>
    </nav>
  );
}

export function TodayView({ inputs, linkFisheries, readiness, fixture, trust, stage }: TodayProps) {
  const { place, now } = inputs;
  const [selected, setSelected] = useState<FisheryId[]>(linkFisheries ?? DEFAULT_FISHERIES);
  const [picking, setPicking] = useState(false);
  const [asking, setAsking] = useState(false);
  // Ask Neritic and alerts cover the pilot ports; elsewhere the actions link out and say so
  const pilot = PILOT_PORTS.some((p) => p.place_id === place.place_id);
  const target = useFactTarget();
  const [sheet, setSheet] = useState<StatusRow | null>(null);
  const [whaleOpen, setWhaleOpen] = useState(false);
  const [saved, setSaved] = useState<{ at: string | null } | null>(null);
  // the same point and periods the server asks for (portToday.ts), so both read one answer
  const conditionsUrl = `/api/conditions?lat=${place.lat.toFixed(4)}&lon=${place.lon.toFixed(4)}&periods=12`;
  const conditions = useConditions(conditionsUrl);

  useEffect(() => {
    writeTodayPlace(place.place_id);
    // the reader's own list (kept on this device) applies unless the link names one
    if (!linkFisheries) {
      const mine = readFisheries();
      if (mine) setSelected(mine);
    }
    const m = savedMark();
    if (m) setSaved({ at: m.at });
  }, [place.place_id, linkFisheries]);

  // every fishery: the facts alerts, Ask Neritic and "what changed" all read (src/pilot/today/portToday.ts builds the same on the server)
  const all = useMemo(() => buildTodayModel({ ...inputs, selected: ALL_FISHERIES, conditions }), [inputs, conditions]);
  const allFacts = all.today.facts;
  // a link to a fact of a fishery the reader does not follow shows that fishery for this visit (nothing is saved)
  const linked = useMemo(() => {
    if (!target) return null;
    const row = all.status.find((r) => r.fact.id === target || r.notices.some((n) => `notice:${n.id}` === target));
    return row && !selected.includes(row.fishery) ? row.fishery : null;
  }, [target, all, selected]);
  const shown = useMemo(() => (linked ? [...selected, linked] : selected), [selected, linked]);
  const model = useMemo(() => buildTodayModel({ ...inputs, selected: shown, conditions }), [inputs, shown, conditions]);

  // arrive on the card a citation named: scroll it into view and mark it for a few seconds
  const marked = useRef<string | null>(null);
  const resettle = useRef(0);
  const seaReady = conditions !== null;
  useEffect(() => {
    if (!target || marked.current === target) return;
    const t = window.setTimeout(() => {
      const el = document.getElementById(factAnchor(target));
      if (!el) return; // a sea fact before the sea has loaded: tried again when it has
      const card = el.closest<HTMLElement>("[data-fact-card]") ?? el;
      marked.current = target;
      document.querySelectorAll("[data-fact-hit]").forEach((x) => {
        x.classList.remove("fact-hit");
        x.removeAttribute("data-fact-hit");
      });
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      // a citation may name a fact inside a closed fold: the fold opens so the reader lands on it, also when
      // it was shut by hand after an earlier visit to the same link (Fold itself reacts only to a hash change)
      window.clearTimeout(resettle.current);
      let inFold = false;
      for (let f = card.closest(".nx-fold"); f; f = f.parentElement?.closest(".nx-fold") ?? null) {
        inFold = true;
        f.previousElementSibling?.querySelector<HTMLButtonElement>('button[aria-expanded="false"]')?.click();
      }
      card.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
      // once a fold's height has settled; a newer citation cancels this, so it never pulls the reader back
      if (inFold) resettle.current = window.setTimeout(() => card.scrollIntoView({ block: "center", behavior: "auto" }), 320);
      void card.offsetWidth; // restart the mark when the same card is reached again
      card.classList.add("fact-hit");
      card.setAttribute("data-fact-hit", target);
    }, 80);
    return () => window.clearTimeout(t);
  }, [target, seaReady, shown]);
  useEffect(() => {
    if (!target) marked.current = null;
  }, [target]);
  useEffect(() => () => window.clearTimeout(resettle.current), []);

  const seaFactIds = useMemo(() => allFacts.filter((f) => f.kind === "weather" || f.kind === "tide" || f.kind === "sea" || f.kind === "observation").map((f) => f.id), [allFacts]);
  const noticeIds = useMemo(() => allFacts.filter((f) => f.kind === "notice").map((f) => f.id), [allFacts]);

  const onSave = useCallback(
    (ids: FisheryId[]) => {
      setSelected(ids);
      writeFisheries(ids);
      setPicking(false);
      const u = new URL(window.location.href);
      u.searchParams.delete("f");
      window.history.replaceState(null, "", u.pathname + u.search);
    },
    [],
  );

  // the trust pill: the dataset is in hand; a clock when any of this page's sources is old
  useReportTrust(trust);
  const oldHere = model.freshness.filter((f) => f.stale);
  useReportSource(oldHere.length ? { id: "evidence", label: "Today's facts", state: "stale", detail: oldHere.map((f) => `${f.label} ${f.when}`).join(" · ") } : null);

  const lede = model.lede;
  const stamp = stableFormat(STAMP, new Date(now));
  const { bind } = useFocusLink();
  const code = Number(place.place_id.replace(/^port-/, ""));
  const port = Number.isFinite(code) ? code : null;
  const marks: StageMark[] = [
    ...model.tests.filter((t) => t.site.lat != null && t.site.lon != null).map((t) => ({ id: `site:${t.site.id}`, lon: t.site.lon!, lat: t.site.lat!, kind: "site" as const })),
    ...(model.sea.buoy?.lat != null && model.sea.buoy.lon != null ? [{ id: `buoy:${model.sea.buoy.id}`, lon: model.sea.buoy.lon, lat: model.sea.buoy.lat, kind: "buoy" as const }] : []),
  ];
  // the notice lines stay; their words stay in the briefing, not across the title and the glance row
  const lines = useMemo(() => (stage?.official ? { ...stage.official, features: stage.official.features.map((f) => ({ ...f, properties: { kind: f.properties?.kind, record_ids: f.properties?.record_ids } })) } : null), [stage]);
  const advisory = model.sea.alerts[0]?.event ?? null;
  const seaLine = lede.sea ? `${lede.sea.replace(/\.$/, "")}${advisory ? ` · ${advisory.toLowerCase()}` : ""}` : null;
  // the hero's one line: at most two of the winds NWS stated, an ellipsis when it stated more (the Sea chapter says them all)
  const heroSea = model.sea.lanes[0] && periodLine(model.sea.lanes[0], 2) ? `${model.sea.lanes[0].name}: ${periodLine(model.sea.lanes[0], 2)}${advisory ? ` · ${advisory.toLowerCase()}` : ""}` : seaLine;
  // on a wide screen the words take the left, so the water and the port stand to the right of them (client only: the camera never renders on the server)
  const [westward] = useState(() => (typeof window !== "undefined" && window.innerWidth >= 1024 ? 0.14 : 0));
  const feedbackFact = model.status[0]?.fact.id ?? null;
  return (
    <div className="env-instrument today min-h-dvh pb-[calc(76px+env(safe-area-inset-bottom,0px))] lg:pb-0" data-testid="today">
      <SkipLink />
      <TopBar active="today" now={new Date(now)} trust={trust} over />
      <main id="main">
        <div className="today-stack">
          <OceanStage
            className="today-stage"
            center={[place.lon - westward, place.lat]}
            zoom={9.3}
            phoneZoom={8.7}
            pitch={45}
            bearing={24}
            field="chlorophyll"
            currents
            data={stage}
            port={port}
            marks={marks}
            lines={lines}
            arrival={place.place_id}
            scrollLinked
          >
            <div className="today-over" data-testid="today-hero">
              {stage?.chlorophyll && (
                <Pop label="Map key" testId="today-map-key" triggerClass="today-key nx-focus" trigger={<><Icon name="info" className="h-[18px] w-[18px]" /><span className="sr-only">Map key</span></>}>
                  <p className="text-[15px] font-semibold text-n-ink">
                    {stage.chlorophyll.caption.what}, {stage.chlorophyll.caption.kind}, {stage.chlorophyll.caption.date}. Not toxin.
                  </p>
                  {stage.chlorophyll.caption.ramp && (
                    <>
                      <div className="mt-2 h-2 rounded-[2px]" style={{ background: stage.chlorophyll.caption.ramp.css }} aria-hidden />
                      <p className="mt-1 flex justify-between font-data text-[12px] text-n-ink-2" aria-label="Scale in milligrams per cubic metre">
                        {stage.chlorophyll.caption.ramp.labels.map((l) => (
                          <span key={l}>{l}</span>
                        ))}
                      </p>
                    </>
                  )}
                  {stage.currents && <p className="mt-2 text-[14px] leading-[20px] text-n-ink-2">Streaks are measured surface currents, {stage.currents.hour}.</p>}
                </Pop>
              )}
              {saved && (
                <p className="today-saved" role="status" data-testid="today-saved-copy">
                  <Icon name="offline" className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>
                    <b className="font-semibold">Saved copy</b>
                    {saved.at ? `, saved ${stableFormat(STAMP, new Date(saved.at))}` : ""}. Status, tests and the sea may have changed since; check again when you have signal.
                  </span>
                </p>
              )}
              <div className="today-date" data-beat="1">
                <span data-testid="today-stamp">{stamp}</span>
                <Freshness items={model.freshness.map((f) => ({ label: f.label, when: f.when ?? "", stale: f.stale, missing: f.missing }))} testId="today-freshness" />
              </div>
              {/* the arrival beat is on a wrapper: on the same element it would replace the scroll-linked condense */}
              <div data-beat="1">
              <h1 className="nx-display nx-display-hero nx-scroll-title today-port" data-testid="today-port">
                <Link href="/today?pick=1" className="today-port-a nx-focus" data-testid="today-change-port" aria-label={`${place.label}: change port`}>
                  {place.label}
                  <svg aria-hidden viewBox="0 0 24 24" className="today-port-c" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                    <path d="m6 9 6 6 6-6" />
                  </svg>
                </Link>
              </h1>
              </div>
              <div data-beat="2">
                <p className="nx-lede today-lede" data-testid="today-lede">
                  {lede.label ? (
                    <>
                      <b className={lede.seasonal ? undefined : sevClass(lede.severity)}>{lede.label}</b>: {lede.text}
                    </>
                  ) : (
                    <span>{lede.text}</span>
                  )}
                  {lede.caution && (
                    <span className="today-old block pt-1 text-[0.8em]" data-testid="lede-caution">
                      {lede.caution}
                    </span>
                  )}
                </p>
                {lede.source && (
                  <div className="today-cue" data-testid="lede-cue">
                    <span>
                      {lede.source.name}
                      {lede.checked_at ? ` · checked ${dayWords(lede.checked_at, now)}` : ""}
                    </span>
                    {lede.verified === true ? null : <Unverified testId="lede-not-verified" />}
                  </div>
                )}
                <div data-fact-card="">
                  <FactAnchors ids={model.ages.map((f) => f.id)} />
                  <p className="today-sealine" data-testid="today-sealine">
                    {heroSea
                      ? heroSea.split(" · ").flatMap((part, i) => [i ? " · " : null, <span key={part}>{part}</span>])
                      : model.sea.state === "loading"
                        ? "Reading the marine forecast…"
                        : "Marine forecast not available."}
                  </p>
                </div>
              </div>
              <GlanceRow model={model} now={now} bind={bind} />
            </div>
          </OceanStage>
          <div className="today-pin" aria-hidden />
        </div>

        <div className="today-brief" data-testid="today-brief">
          <div className="today-brief-in">
            <WhatChanged place={place.place_id} facts={allFacts} ready={conditions !== null} now={now} />
            <StatusStrip rows={model.status} statusError={model.statusError} whale={model.whale} otherNotices={noticeIds} onOpen={setSheet} onOpenWhale={() => setWhaleOpen(true)} onEdit={() => setPicking(true)} />
            <SeaWeather sea={model.sea} now={now} seaLine={seaLine} place={place.place_id} factIds={seaFactIds} />
            <ToxinTests tests={model.tests} error={model.testsError} now={now} sourceUrl={inputs.biotoxin?.source.url ?? null} />
            <BloomOutlook bloom={model.bloom} now={now} label={place.label} exploreHref={`/?view=map&sel=harbor:${place.place_id.replace(/^port-/, "")}`} />
            <Actions place={place.place_id} label={place.label} conditionsUrl={conditionsUrl} readiness={readiness} pilot={pilot} onAsk={() => setAsking(true)} feedbackFact={feedbackFact} />
            <PilotFooter />
          </div>
        </div>
      </main>
      <PilotDock active="today" />
      {/* the device's alert check, on the same facts /account and the fan-out compare */}
      <AlertCheck todays={[all.today]} sample={fixture} />
      {pilot && <AskSheet place={place.place_id} open={asking} onClose={() => setAsking(false)} />}
      <StatusSheet row={sheet} whale={model.whale} now={now} onClose={() => setSheet(null)} />
      <WhaleSheet open={whaleOpen} whale={model.whale} now={now} onClose={() => setWhaleOpen(false)} />
      <FisheryPicker open={picking} selected={selected} onClose={() => setPicking(false)} onSave={onSave} />
    </div>
  );
}
