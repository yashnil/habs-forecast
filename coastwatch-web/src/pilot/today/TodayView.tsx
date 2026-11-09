"use client";

import "./today.css";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PILOT_PORTS, type FisheryId, type Readiness } from "@/pilot/contracts";
import type { Conditions } from "@/lib/conditions";
import { DEFAULT_FISHERIES, FISHERY_OPTIONS, buildTodayModel, dayWords, nextTide, statusTally, type PortTodayInputs, type StatusRow } from "@/pilot/facts";
import { TopBar } from "@/v4/shell/TopBar";
import { PilotDock, PilotFooter } from "@/pilot/shell/PilotShell";
import { SkipLink } from "@/v4/shell/SkipLink";
import { readDaylight, setDaylight } from "@/v4/shell/daylight";
import { Icon } from "@/v4/ui/Icon";
import { readFisheries, writeFisheries, writeTodayPlace } from "./prefs";
import { savedMark } from "./offline";
import { FisheryPicker, StatusSheet, StatusStrip } from "./Status";
import { SeaWeather } from "./Sea";
import { ToxinTests } from "./Tests";
import { BloomOutlook, WhaleRules } from "./Bloom";
import { WhatChanged } from "./Changed";
import { Actions } from "./Actions";
import { FactAnchors, factAnchor, sevClass } from "./ui";
import { AlertCheck } from "@/pilot/alerts/AlertCheck";
import { AskSheet } from "@/pilot/ask/ui/AskSheet";
import { periodLine } from "./marine";

export type TodayProps = {
  inputs: Omit<PortTodayInputs, "conditions" | "selected">;
  /** from the link (?f=…); null: the reader's saved list, else the default */
  linkFisheries: FisheryId[] | null;
  readiness: { alerts: Readiness; ask: Readiness; sms: Readiness };
  /** the server is reading the test fixture: alerts made here are marked as samples */
  fixture: boolean;
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

function SunToggle() {
  const [day, setDay] = useState(false);
  useEffect(() => setDay(readDaylight()), []);
  return (
    <button
      type="button"
      aria-pressed={day}
      onClick={() => {
        setDaylight(!day);
        setDay(!day);
      }}
      className="today-btn nx-focus nx-press nx-tint"
      data-testid="today-sun"
      title="High-contrast colours for bright sun"
    >
      <Icon name="sun" className="h-[18px] w-[18px]" /> {day ? "Night colours" : "Bright sun"}
    </button>
  );
}

/** Desktop only: the page at a glance, beside the answer (each line links to its section). */
function Glance({ model, now }: { model: ReturnType<typeof buildTodayModel>; now: string }) {
  const first = model.sea.lanes[0];
  const tide = nextTide(model.sea.tides, now);
  const t = model.tests[0];
  const rows: [string, string, string][] = [
    ["Status", statusTally(model.status) || "not available", "#status"],
    ["Sea", first ? `${first.name}: ${periodLine(first) ?? "see NWS"}` : model.sea.state === "loading" ? "loading…" : "not available", "#sea"],
    ["Next tide", tide ? `${tide.type === "H" ? "High" : "Low"} ${tide.ft.toFixed(1)} ft at ${((h, m) => `${h % 12 || 12}:${m} ${h < 12 ? "AM" : "PM"}`)(Number(tide.at.slice(11, 13)), tide.at.slice(14, 16))}` : model.sea.state === "loading" ? "loading…" : "not available", "#sea"],
    ["Latest test", t?.latest ? `${t.site.species}, ${dayWords(t.latest.date, now)}${t.latest.avg_ppm != null ? ` · avg ${t.latest.avg_ppm} ppm` : ""}` : model.testsError ? "not available" : "none published here", "#tests"],
    ["C-HARM", model.bloom.charmFact.value ? `${model.bloom.charmFact.value} (run ${model.bloom.charm?.issued ? dayWords(model.bloom.charm.issued, now) : "?"})` : "no value here", "#bloom"],
    ["Whale rules", model.whale ? `Zone ${model.whale.zone}: ${(model.whale.action ?? "none on record").split(";")[0].replace(/\.$/, "")}` : "not available", "#whale"],
  ];
  return (
    <aside aria-label="At a glance" className="hidden self-end lg:block" data-testid="today-glance">
      <p className="nx-caps">At a glance</p>
      <dl className="mt-2 border-t border-n-line">
        {rows.map(([k, v, href]) => (
          <div key={k} className="grid grid-cols-[96px_minmax(0,1fr)] gap-3 border-b border-n-line py-2.5 text-[14px] leading-snug">
            <dt className="text-n-ink-3">{k}</dt>
            <dd>
              <a href={href} className="nx-focus text-n-ink hover:underline">
                {v}
              </a>
            </dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}

export function TodayView({ inputs, linkFisheries, readiness, fixture }: TodayProps) {
  const { place, now } = inputs;
  const [selected, setSelected] = useState<FisheryId[]>(linkFisheries ?? DEFAULT_FISHERIES);
  const [picking, setPicking] = useState(false);
  const [asking, setAsking] = useState(false);
  // Ask Neritic and alerts cover the pilot ports; elsewhere the actions link out and say so
  const pilot = PILOT_PORTS.some((p) => p.place_id === place.place_id);
  const target = useFactTarget();
  const [sheet, setSheet] = useState<StatusRow | null>(null);
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
      card.scrollIntoView({ block: "center", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      void card.offsetWidth; // restart the mark when the same card is reached again
      card.classList.add("fact-hit");
      card.setAttribute("data-fact-hit", target);
    }, 80);
    return () => window.clearTimeout(t);
  }, [target, seaReady, shown]);
  useEffect(() => {
    if (!target) marked.current = null;
  }, [target]);

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

  const lede = model.lede;
  const stamp = STAMP.format(new Date(now));
  return (
    <div className="env-instrument today min-h-dvh pb-[calc(76px+env(safe-area-inset-bottom,0px))] lg:pb-10" data-testid="today">
      <SkipLink />
      <TopBar active="today" now={new Date(now)} />
      <main id="main">
        <header className="today-hero" data-testid="today-hero">
          <svg className="today-hero-iso" viewBox="0 0 400 200" preserveAspectRatio="none" aria-hidden="true">
            <path d="M0 150 C80 128 120 172 200 142 S330 100 400 120" />
            <path d="M0 172 C90 152 140 190 220 166 S340 132 400 148" />
            <path d="M0 128 C70 104 130 146 190 118 S320 74 400 92" className="index" />
            <path d="M0 106 C60 82 120 120 180 96 S310 52 400 66" />
            <path d="M0 194 C100 178 150 206 230 188 S350 160 400 172" />
          </svg>
          <div className="today-wrap relative grid gap-x-12 pb-6 pt-5 md:pb-9 md:pt-8 lg:grid-cols-[minmax(0,1fr)_340px]">
            <div className="min-w-0">
            {saved && (
              <p className="mb-4 flex items-start gap-2 rounded-[4px] border border-[var(--t-amber)] px-3 py-2 text-[14px] text-n-ink" role="status" data-testid="today-saved-copy">
                <Icon name="offline" className="mt-0.5 h-4 w-4 t-amber" />
                <span>
                  <b className="font-semibold">Saved copy</b>
                  {saved.at ? `, saved ${STAMP.format(new Date(saved.at))}` : ""}. Status, tests and the sea may have changed since; check again when you have signal.
                </span>
              </p>
            )}
            <p className="nx-caps">
              Today at your port · <span data-testid="today-stamp">{stamp}</span>
            </p>
            <h1 className="today-port mt-2 text-n-ink" data-testid="today-port">
              {place.label}
            </h1>
            <p className="today-lede mt-3 text-n-ink" data-testid="today-lede">
              {lede.label ? (
                <>
                  <b className={lede.seasonal ? undefined : sevClass(lede.severity)}>{lede.label}</b>: {lede.text}
                </>
              ) : (
                <span className={lede.severity === "amber" ? "t-amber" : undefined}>{lede.text}</span>
              )}
              {lede.more > 0 && (
                <>
                  {" "}
                  <a href="#status" className="nx-focus whitespace-nowrap underline decoration-n-line-strong underline-offset-4">
                    {lede.more} more below.
                  </a>
                </>
              )}
              {lede.caution && (
                <span className="t-amber block pt-1" data-testid="lede-caution">
                  {lede.caution}
                </span>
              )}
              {lede.sea && <span className="block pt-1 text-n-ink-2">{lede.sea}</span>}
            </p>
            <div className="mt-4" data-fact-card="">
            <FactAnchors ids={model.ages.map((f) => f.id)} />
            <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[13.5px] text-n-ink-2" aria-label="How fresh the data are" data-testid="today-freshness">
              {model.freshness.map((f) => (
                <li key={f.label} className={f.stale ? "t-amber" : undefined}>
                  <span className="text-n-ink-3">{f.label}</span> {f.when}
                  {f.stale && !f.missing ? " (old)" : ""}
                </li>
              ))}
            </ul>
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <Link href="/today?pick=1" className="today-btn nx-focus nx-press nx-tint" data-testid="today-change-port">
                <Icon name="harbor" className="h-[18px] w-[18px]" /> Change port
              </Link>
              <SunToggle />
              <a
                href={`/ask?place=${encodeURIComponent(place.place_id)}`}
                onClick={(e) => {
                  if (!pilot) return;
                  e.preventDefault();
                  setAsking(true);
                }}
                aria-haspopup={pilot ? "dialog" : undefined}
                className="today-btn nx-focus nx-press nx-tint"
                data-testid="hero-ask"
              >
                <Icon name="ask" className="h-[18px] w-[18px]" /> Ask
              </a>
              <Link href="/about" className="nx-focus ml-1 inline-flex min-h-11 items-center text-[13.5px] text-n-ink-3 underline decoration-n-line underline-offset-4">
                Research preview · not an official source
              </Link>
            </div>
            </div>
            <Glance model={model} now={now} />
          </div>
        </header>

        <div className="today-wrap">
          <div className="today-grid" data-layout={model.testsError || !model.tests.length ? (model.statusError ? "sparse" : "notests") : undefined}>
            <StatusStrip rows={model.status} now={now} place={place.place_id} statusError={model.statusError} otherNotices={noticeIds} onOpen={setSheet} onEdit={() => setPicking(true)} />
            <SeaWeather sea={model.sea} now={now} lede={lede.sea} place={place.place_id} factIds={seaFactIds} />
            <ToxinTests tests={model.tests} error={model.testsError} now={now} sourceUrl={inputs.biotoxin?.source.url ?? null} place={place.place_id} reveal={target} />
            <BloomOutlook bloom={model.bloom} now={now} label={place.label} place={place.place_id} />
            <WhaleRules whale={model.whale} now={now} place={place.place_id} />
            <WhatChanged place={place.place_id} facts={allFacts} ready={conditions !== null} now={now} />
            <Actions place={place.place_id} label={place.label} conditionsUrl={conditionsUrl} readiness={readiness} pilot={pilot} onAsk={() => setAsking(true)} />
          </div>
          <PilotFooter />
        </div>
      </main>
      <PilotDock active="today" />
      {/* the device's alert check, on the same facts /account and the fan-out compare */}
      <AlertCheck todays={[all.today]} sample={fixture} />
      {pilot && <AskSheet place={place.place_id} open={asking} onClose={() => setAsking(false)} />}
      <StatusSheet row={sheet} now={now} onClose={() => setSheet(null)} />
      <FisheryPicker open={picking} selected={selected} onClose={() => setPicking(false)} onSave={onSave} />
    </div>
  );
}
