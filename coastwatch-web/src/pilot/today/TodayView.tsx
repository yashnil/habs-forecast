"use client";

import "./today.css";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { FisheryId, Readiness } from "@/pilot/contracts";
import type { Conditions } from "@/lib/conditions";
import { DEFAULT_FISHERIES, FISHERY_OPTIONS, buildTodayModel, dayWords, nextTide, type PortTodayInputs, type StatusRow } from "@/pilot/facts";
import { TopBar, NavTabs } from "@/v4/shell/TopBar";
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
import { sevClass } from "./ui";
import { periodLine } from "./marine";

export type TodayProps = {
  inputs: Omit<PortTodayInputs, "conditions" | "selected">;
  /** from the link (?f=…); null: the reader's saved list, else the default */
  linkFisheries: FisheryId[] | null;
  readiness: { alerts: Readiness; ask: Readiness };
};

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
  const n = (sev: string) => model.status.filter((r) => r.severity === sev).length;
  const first = model.sea.lanes[0];
  const tide = nextTide(model.sea.tides, now);
  const t = model.tests[0];
  const rows: [string, string, string][] = [
    ["Status", [n("red") && `${n("red")} closed or delayed`, n("amber") && `${n("amber")} restricted`, n("neutral") && `${n("neutral")} none on record`].filter(Boolean).join(" · ") || "not available", "#status"],
    ["Sea", first ? `${first.name}: ${periodLine(first) ?? "see NWS"}` : model.sea.state === "loading" ? "loading…" : "not available", "#sea"],
    ["Next tide", tide ? `${tide.type === "H" ? "High" : "Low"} ${tide.ft.toFixed(1)} ft at ${((h, m) => `${h % 12 || 12}:${m} ${h < 12 ? "AM" : "PM"}`)(Number(tide.at.slice(11, 13)), tide.at.slice(14, 16))}` : model.sea.state === "loading" ? "loading…" : "not available", "#sea"],
    ["Latest test", t?.latest ? `${t.site.species}, ${dayWords(t.latest.date, now)}${t.latest.avg_ppm != null ? ` · avg ${t.latest.avg_ppm} ppm` : ""}` : "none published here", "#tests"],
    ["C-HARM", model.bloom.charmFact.value ? `${model.bloom.charmFact.value} (run ${model.bloom.charm?.issued ? dayWords(model.bloom.charm.issued, now) : "?"})` : "no value here", "#bloom"],
    ["Whale rules", model.whale ? `Zone ${model.whale.zone}: ${model.whale.action ?? "none on record"}` : "no record", "#whale"],
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

export function TodayView({ inputs, linkFisheries, readiness }: TodayProps) {
  const { place, now } = inputs;
  const [selected, setSelected] = useState<FisheryId[]>(linkFisheries ?? DEFAULT_FISHERIES);
  const [picking, setPicking] = useState(false);
  const [sheet, setSheet] = useState<StatusRow | null>(null);
  const [saved, setSaved] = useState<{ at: string | null } | null>(null);
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

  const model = useMemo(() => buildTodayModel({ ...inputs, selected, conditions }), [inputs, selected, conditions]);
  // "what changed" compares every fishery, so changing the list is never reported as news
  const allFacts = useMemo(() => buildTodayModel({ ...inputs, selected: FISHERY_OPTIONS.map((o) => o.id), conditions }).today.facts, [inputs, conditions]);

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
                  <b className={sevClass(lede.severity)}>{lede.label}</b>: {lede.text}
                </>
              ) : (
                <span className={lede.severity === "amber" ? "t-amber" : undefined}>{lede.text}</span>
              )}
              {lede.more > 0 && (
                <>
                  {" "}
                  <a href="#status" className="nx-focus underline decoration-n-line-strong underline-offset-4">
                    {lede.more} more {lede.more === 1 ? "restriction" : "restrictions"} below.
                  </a>
                </>
              )}
              {lede.sea && <span className="block pt-1 text-n-ink-2">{lede.sea}</span>}
            </p>
            <ul className="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-[13.5px] text-n-ink-2" aria-label="How fresh the data are" data-testid="today-freshness">
              {model.freshness.map((f) => (
                <li key={f.label} className={f.stale ? "t-amber" : undefined}>
                  <span className="text-n-ink-3">{f.label}</span> {f.when}
                  {f.stale ? " (old)" : ""}
                </li>
              ))}
            </ul>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <Link href="/today?pick=1" className="today-btn nx-focus nx-press nx-tint" data-testid="today-change-port">
                <Icon name="harbor" className="h-[18px] w-[18px]" /> Change port
              </Link>
              <SunToggle />
              <Link href="/about" className="nx-focus ml-1 inline-flex min-h-11 items-center text-[13.5px] text-n-ink-3 underline decoration-n-line underline-offset-4">
                Research preview · not an official source
              </Link>
            </div>
            </div>
            <Glance model={model} now={now} />
          </div>
        </header>

        <div className="today-wrap">
          <div className="today-grid">
            <StatusStrip rows={model.status} now={now} statusError={model.statusError} onOpen={setSheet} onEdit={() => setPicking(true)} />
            <SeaWeather sea={model.sea} now={now} lede={lede.sea} />
            <ToxinTests tests={model.tests} error={model.testsError} now={now} sourceUrl={inputs.biotoxin?.source.url ?? null} />
            <BloomOutlook bloom={model.bloom} now={now} label={place.label} />
            <WhaleRules whale={model.whale} now={now} />
            <WhatChanged place={place.place_id} facts={allFacts} ready={conditions !== null} now={now} />
            <Actions place={place.place_id} label={place.label} conditionsUrl={conditionsUrl} readiness={readiness} />
          </div>
          <footer className="mt-10 border-t border-n-line py-6 text-[13px] leading-relaxed text-n-ink-3">
            Neritic is an independent research preview, not an official source and not a seafood-safety service. Closures, seasons and advisories come from CDFW and CDPH; their notices always take precedence. Chlorophyll does not measure toxin. Forecasts are probabilities, not guarantees.
          </footer>
        </div>
      </main>
      <div className="env-instrument fixed inset-x-0 bottom-0 z-40 border-t border-n-line bg-n-surface lg:hidden" style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
        <NavTabs active="today" size="dock" />
      </div>
      <StatusSheet row={sheet} now={now} onClose={() => setSheet(null)} />
      <FisheryPicker open={picking} selected={selected} onClose={() => setPicking(false)} onSave={onSave} />
    </div>
  );
}
