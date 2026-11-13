"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Micro, Pass, RollingNumber, usePhone } from "@/nx/ds";
import type { Place } from "@/nx/ds";
import { tripFishery, type CoastFishery } from "./fisheries";
import { breakEvenGross, fieldDomId, needsWords, nextWords, passSteps, whatIfRows } from "./pass";
import { readShare } from "./scenarios";
import { TripWaterfall, TripWhatIf } from "./TripFigures";
import { TripEquation, TripMore, TripTerms } from "./TripInputs";
import { TripWorked, useTripPreview } from "./TripResults";
import { useTripScenarios } from "./TripScenarios";
import { useTripWorkspace } from "./useTripWorkspace";
import { breakEvenCatchText, money, read, setCatch, waitingWords, type FieldId, type Step } from "./workspace";
import s from "./trip.module.css";

const num = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 2 });

/**
 * The trip (v14 synthesis, "trip-empty" and "trip-done"): one perforated pass along the foot of the chart. Its main part
 * is the hypothetical-trip strip (port, fishery, the season line in the record's words), the trip as an equation, its
 * terms, and under them the waterfall from gross to net with the what-if dumbbells beside it. The stub holds the one
 * result (net to boat, rolling per digit), what it is not, and the one filled button. Under the main part's fold:
 * the rest of the inputs, every worked figure, the grid and one-at-a-time, and the saved trips.
 *
 * On a phone the pass is a block in the sheet: the stub first, then the strip, the waterfall and the equation, with
 * the result and the one button kept in a bar at the sheet's foot. All of it is the headless model in
 * ./useTripWorkspace; this file only lays it out.
 */
export function TripWorkspace({ fishery, placeKey, at, onFishery, now, strip, toBrief }: { fishery: CoastFishery | null; placeKey: string; at: Place | null; onFishery: (id: string) => void; now: Date | null; strip: ReactNode; toBrief: () => void }) {
  const trip = useTripWorkspace({ placeKey, fishery, onFishery });
  const look = useTripPreview(trip.result, trip.preview);
  const phone = usePhone();
  const { ready, edit, ws, result } = trip;
  const unit = fishery?.unit ?? "lb";

  // a numbers-only link (#trip=…): its numbers become the worksheet, for the fishery its query names
  useEffect(() => {
    if (!ready || !/[#&]trip=/.test(window.location.hash)) return;
    const want = new URLSearchParams(window.location.search).get("fishery");
    if (want && tripFishery(want) && want !== fishery?.id) {
      onFishery(want);
      return;
    }
    const shared = readShare(window.location.hash, fishery?.id ?? null);
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
    if (shared) edit(() => shared);
  }, [ready, edit, fishery, onFishery]);

  // the one button walks the blanks: it takes the reader to the next field the trip waits for
  const scroller = useRef<HTMLDivElement>(null);
  const go = useCallback((id: FieldId) => {
    const el = document.getElementById(fieldDomId(id));
    const target = el && (el.matches("input, [role=radio]") ? el : el.querySelector<HTMLElement>("[role=radio][tabindex='0'], input"));
    if (target) {
      target.scrollIntoView({ block: "nearest", behavior: "smooth" });
      target.focus({ preventScroll: true });
      return;
    }
    // a field under the fold (fuel from distance, a second line): the first empty input there
    const more = document.getElementById("trip-more")?.parentElement;
    const empty = more ? [...more.querySelectorAll<HTMLInputElement>("input")].find((i) => !i.value) : null;
    (empty ?? more)?.scrollIntoView({ block: "center", behavior: "smooth" });
    empty?.focus({ preventScroll: true });
  }, []);
  const waiting = ready ? result.waiting : [];
  const nextId = waiting[0] ?? null;
  const next = nextId ? { words: nextWords(nextId, ws, fishery), go: () => go(nextId) } : null;
  const toMore = useCallback(() => document.getElementById("trip-more")?.scrollIntoView({ block: "start", behavior: "smooth" }), []);
  const saving = useTripScenarios({ trip, fishery, placeKey, next, toBrief, toMore });

  // a what-if end sets that input, and the field it set pops once
  const [pops, setPops] = useState<Record<string, number>>({});
  const apply = useCallback(
    (k: "catch" | "price", v: number) => {
      const id = k === "catch" ? "catch.0.amount" : "catch.0.price";
      edit((w) => setCatch(w, 0, k === "catch" ? { amount: num(v).replace(/,/g, "") } : { price: v.toFixed(2) }));
      setPops((p) => ({ ...p, [id]: (p[id] ?? 0) + 1 }));
    },
    [edit],
  );

  const shown = look.shown;
  const rec = shown.audience === "recreational";
  const main = rec ? shown.out.operating : shown.out.net;
  const steps = useMemo(() => passSteps(shown.steps), [shown.steps]);
  const { n } = useMemo(() => read(ws, fishery), [ws, fishery]);
  const basis = useCallback(
    (st: Step): string => {
      const c = n.catch[0];
      if (st.id === "gross") return ws.catch.length === 1 && c?.amount != null && c.price != null ? `${num(c.amount)} ${unit} × ${money(c.price)}` : "catch × price";
      if (st.id === "fuel") return result.out.gallons.status === "value" && n.fuelPrice != null ? `${num(result.out.gallons.value)} gal × ${money(n.fuelPrice)}` : "gal × $/gal";
      if (st.id === "fees") return n.feesPct != null ? `${num(n.feesPct)}% of gross${ws.feesPerUnit !== null ? " + per lb" : ""}` : "% of gross";
      if (st.id === "crew") {
        if (n.crew === 0) return "none paid";
        if (ws.crewKind === "dayRate") return n.crew != null && n.dayRate != null ? `${n.crew} × ${money(n.dayRate)} × ${num(n.days ?? 0)} d` : "crew × day rate";
        return n.crew != null && n.totalPct != null ? `${n.crew} × ${num(n.totalPct / Math.max(1, n.crew))}%` : "crew × share";
      }
      if (st.id.startsWith("cost-")) return st.amount === null ? "blank" : "";
      return "";
    },
    [n, ws, unit, result.out.gallons],
  );
  const be = useMemo(() => breakEvenGross(ws, fishery, shown), [ws, fishery, shown]);
  // the what-if is always the reader's own trip (a grid scenario being looked at does not move it, nor the page under the hand)
  const rows = useMemo(() => whatIfRows(ws, fishery, result), [ws, fishery, result]);
  const missing = result.steps.filter((st) => st.amount === null);
  const stillWaiting =
    missing.length && missing.length < result.steps.length && result.waiting.length
      ? `${missing.length === 1 ? `${missing[0].label} is` : `${missing.length} bars are`} not drawn yet: still ${waitingWords({ status: "waiting", fields: result.waiting }, ws, fishery)}.`
      : null;

  // the main part keeps the front (strip, equation, figures) in view; what is under the fold scrolls inside it, and the
  // pass never takes more than 55 % of the screen's height (the chart keeps at least 45 %)
  const front = useRef<HTMLDivElement>(null);
  const stripBox = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = front.current, top = stripBox.current, sc = scroller.current;
    if (!el || !top || !sc || typeof ResizeObserver === "undefined") return;
    const set = () => sc.style.setProperty("--front-h", `${el.offsetHeight + top.offsetHeight}px`);
    set();
    const ro = new ResizeObserver(set);
    ro.observe(el);
    ro.observe(top);
    return () => ro.disconnect();
  }, []);

  // phone: the bar at the sheet's foot repeats the result only once the stub's figure has scrolled away
  const heroRef = useRef<HTMLElement>(null);
  const [heroSeen, setHeroSeen] = useState(true);
  useEffect(() => {
    const el = heroRef.current;
    if (!phone || !el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setHeroSeen(e.isIntersecting), { threshold: 0.5 });
    io.observe(el);
    return () => io.disconnect();
  }, [phone]);
  const label = rec ? "Trip cost" : "Net to boat";
  const figure = (
    <div className={s.net} data-testid="trip-net">
      <span className={s.vh}>{label}, hypothetical: </span>
      {main.status === "value" ? (
        <>
          <span className={s.vh} data-figure-exact>
            {money(main.value)}
          </span>
          <span className={s.netFig} aria-hidden="true" data-testid="trip-net-figure" data-loss={main.value < 0 || undefined}>
            <RollingNumber value={main.value} format={money} cents />
          </span>
        </>
      ) : (
        <>
          <span className={s.netSlot} aria-hidden="true">
            $ —
          </span>
          <span className={s.vh}>{main.status === "waiting" ? waitingWords(main, ws, fishery) : main.status === "none" ? main.reason : "does not apply"}</span>
        </>
      )}
    </div>
  );
  const note = main.status === "value" || !waiting.length ? (rec ? "What the trip costs. A recreational catch is not sold: there is no price, gross or net." : "Before insurance, permits, upkeep, loans and your own pay. Not profit or take-home.") : needsWords(waiting, ws, fishery);
  const hero = (
    <section ref={heroRef} className={s.hero} aria-label={`${label}, hypothetical, from your numbers`} data-testid="trip-hero">
      <Micro glyph="hypothetical" className={s.heroMicro}>
        Hypothetical · {rec ? "trip cost" : "net to boat"}
      </Micro>
      {figure}
      <p className={s.showing} aria-live="polite" data-testid="trip-showing">
        {look.words}
      </p>
      <p className={s.basis} data-testid="trip-note">
        {note}
      </p>
    </section>
  );

  // phone: the bar at the sheet's foot keeps the result and the one button in reach
  const be0 = shown.out.breakEvenCatch;
  const second = look.words ? look.words.replace("Showing ", "With ").replace("%.", "%") : rec ? "Hypothetical" : be0.status === "value" ? `Break-even ${breakEvenCatchText(be0.value, unit)}` : "Hypothetical";
  const netBar = (
    <div className={s.netBar} data-testid="trip-netbar">
      <p className={s.netBarText} data-quiet={heroSeen || undefined}>
        <span className={s.netBarLabel}>{label}</span> <strong data-testid="trip-netbar-value">{main.status === "value" ? money(main.value) : "waiting"}</strong>
        <span className={s.netBarShowing}>{second}</span>
      </p>
      {saving.primary}
    </div>
  );

  const stub = (
    <>
      {hero}
      {phone ? null : <div className={s.cta}>{saving.primary}</div>}
      {ready ? saving.bar : null}
    </>
  );

  return (
    <Pass label={`Hypothetical trip${fishery ? `, ${fishery.label}` : ""}`} at={at} stub={stub} stubWidth={320} className={s.pass} stubClassName={s.stub} data-testid="trip-pass">
      <div className={s.scroll} ref={scroller}>
        <div ref={stripBox}>{strip}</div>
        <div className={s.front} ref={front}>
          <div className={s.inputs} data-testid="trip-inputs" role="region" aria-label="Trip inputs">
            <TripEquation ws={ws} edit={edit} result={result} fishery={fishery} pops={pops} />
            <TripTerms
              ws={ws}
              edit={edit}
              fishery={fishery}
            />
          </div>
          <div className={s.figs} data-rows={rows ? "" : undefined}>
            <TripWaterfall steps={steps} basis={basis} rec={rec} breakEven={be} label={rec ? "Fuel and costs up to the trip cost, hypothetical, from your numbers" : "Gross to net to boat, hypothetical, from your numbers"} description={stillWaiting} />
            {rows && result.out.net.status === "value" ? <TripWhatIf rows={rows} now={result.out.net.value} onApply={apply} /> : null}
          </div>
        </div>
        <TripMore ws={ws} edit={edit} result={result} fishery={fishery} now={now} />
        <TripWorked ws={ws} result={result} look={look} fishery={fishery} />
        {saving.more}
      </div>
      {phone ? netBar : null}
    </Pass>
  );
}
