"use client";

import { useEffect, useRef } from "react";
import { Button, DeskAside, DeskBody, DeskMain, DeskSticky } from "@/nx/ds";
import type { TripFlag } from "./context";
import { tripFishery, type CoastFishery } from "./fisheries";
import { readShare } from "./scenarios";
import { TripInputs } from "./TripInputs";
import { TripResults, useTripPreview } from "./TripResults";
import { TripScenarios } from "./TripScenarios";
import { useTripWorkspace } from "./useTripWorkspace";
import { breakEvenCatchText, money } from "./workspace";
import s from "./trip.module.css";

/**
 * The trip workspace (v13): inputs in the aside, the result and what it is made of in the main, both in view
 * together on a wide screen. On a phone the page scrolls inputs first and a bar above the destinations keeps
 * the result (and any previewed scenario) in reach. All of it is the headless model in ./useTripWorkspace;
 * this file only lays it out.
 */
export function TripWorkspace({ fishery, placeKey, onFishery, now, flag }: { fishery: CoastFishery | null; placeKey: string; onFishery: (id: string) => void; now: Date | null; flag: TripFlag | null }) {
  const trip = useTripWorkspace({ placeKey, fishery, onFishery });
  const look = useTripPreview(trip.result, trip.preview);
  const hero = useRef<HTMLDivElement>(null);

  // a numbers-only link (#trip=…): its numbers become the worksheet, for the fishery its query names
  const { ready, edit } = trip;
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

  const rec = look.shown.audience === "recreational";
  const main = rec ? look.shown.out.operating : look.shown.out.net;
  const be = look.shown.out.breakEvenCatch;
  const each = look.shown.out.perPerson;
  // the bar's second line: the scenario being looked at, else the break-even (or the split), else that it is hypothetical
  const second = look.words
    ? look.words.replace("Showing ", "With ").replace("%.", "%")
    : rec
      ? each.status === "value"
        ? `${money(each.value)} each`
        : "Hypothetical"
      : be.status === "value"
        ? `Break-even ${breakEvenCatchText(be.value, fishery?.unit ?? "lb")}`
        : "Hypothetical";
  return (
    <>
      <DeskBody layout="main-aside" aside="start">
        <DeskAside label="Trip inputs" sticky className={s.aside}>
          <h2 className={s.h2}>Your numbers</h2>
          <p className={s.sub}>Nothing is filled in. A blank is a gap, never 0.</p>
          <TripInputs ws={trip.ws} edit={trip.edit} result={trip.result} fishery={fishery} now={now} />
        </DeskAside>
        <DeskMain>
          <div ref={hero} className={s.anchor} />
          <TripResults ws={trip.ws} result={trip.result} look={look} fishery={fishery} flag={flag} strip={<TripScenarios trip={trip} fishery={fishery} placeKey={placeKey} />} />
        </DeskMain>
      </DeskBody>
      <DeskSticky className={s.netBar} data-testid="trip-netbar">
        <p className={s.netBarText}>
          <span>
            <span className={s.netBarLabel}>{rec ? "Trip cost" : "Net to boat"}</span> <strong data-testid="trip-netbar-value">{main.status === "value" ? money(main.value) : "waiting"}</strong>
          </span>
          <span className={s.netBarShowing}>{second}</span>
        </p>
        <Button variant="quiet" onClick={() => hero.current?.scrollIntoView({ block: "start", behavior: "smooth" })}>
          Results
        </Button>
      </DeskSticky>
    </>
  );
}
