"use client";

import { LayoutGroup, motion, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Dock, Drawer, Glyph, Micro, PageLayout, Popover, Ticket, TicketMain, TicketStub, usePhone } from "@/nx/ds";
import { day, dayTime, time } from "@/nx/ds/format";
import { reflow } from "@/nx/ds/motion";
import { usePorts } from "@/nx/shell/port";
import { PlanFields } from "@/nx/coast/PlanFields";
import { EMPTY_PLAN, type PlanDraft } from "@/nx/coast/plan";
import { departure, limitValue, type NextAction as Next } from "../lifecycle";
import type { TripModel } from "../useTrip";
import { NextAction } from "./NextAction";
import { TripHeading } from "./TripHeading";
import type { KnownView } from "../known";
import type { OperatingLimits, Snapshot, Trip } from "../types";
import { tripFishery } from "@/nx/coast/fisheries";
import { breakEvenCatchText, evaluateWorkspace, fromDraft, money as fmtMoney } from "@/nx/coast/workspace";
import type { AdvisoryIn } from "../windows";
import { DepartureTimeline } from "./DepartureTimeline";
import { Ledger } from "./Ledger";
import { OutcomeSheet } from "./OutcomeForm";
import { PlannedVsActual } from "./PlannedVsActual";
import { TripChart } from "./TripChart";
import { ChangeFeed } from "./ChangeFeed";
import { RecheckPanel } from "./RecheckPanel";
import { SnapshotView } from "./SnapshotView";
import { timelineOf } from "../feed";
import { tripsStore } from "../useTripsData";
import s from "./trips.module.css";

type Sheet = null | "changes" | "snapshot" | "edit" | "outcome" | "money";

/**
 * The Trip Workspace (v15 §5): where, what, when, what changed, what still needs checking, on the chart.
 *
 * Wide screen: one ticket beside the port. Its main part is the heading and the one dominant "Since you planned this
 * trip" with the next action under it; its stub is the quieter row (the fishery's official line from the planning
 * snapshot, its verified state as a glyph; the compact ledger; the device-local footnote). The departure timeline is
 * the dock along the chart's foot, the showpiece. The chart itself is the trip's geography (TripChart). Evidence opens
 * on demand in drawers. Phone: heading, changes, the next action pinned at the sheet's foot, the timeline, then the
 * rest. A completed trip leads with planned vs actual; a cancelled one is quiet and reopenable.
 */
export function Workspace({ model }: { model: TripModel }) {
  const { trip, planning, latest, recheck, next, busy } = model;
  const phone = usePhone();
  const reduce = useReducedMotion();
  const ports = usePorts();
  const port = trip ? (ports.find((p) => p.key === trip.placeKey) ?? null) : null;
  const [sheet, setSheet] = useState<Sheet>(null);
  const [now, setNow] = useState<Date | null>(null);
  const clock = model.clock;
  useEffect(() => setNow(clock()), [clock]);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const [more, setMore] = useState(false);
  const [invite, setInvite] = useState(false);
  const nowIso = now?.toISOString() ?? planning?.takenAt ?? "";
  // the change history: planning, each kept recheck, superseded planning snapshots (D's feed.ts)
  const timeline = useMemo(() => {
    if (!trip || !planning || !nowIso) return [];
    const get = (ids: string[]) => ids.map((i) => tripsStore().getSnapshot(i)).filter((x): x is Snapshot => !!x);
    return timelineOf(planning, get(trip.snapshots.rechecks), get(trip.snapshots.superseded), nowIso);
  }, [trip, planning, nowIso]);
  const superseded = useMemo(() => (trip ? trip.snapshots.superseded.map((i) => tripsStore().getSnapshot(i)).filter((x): x is Snapshot => !!x) : []), [trip]);

  const act = useCallback(
    (n: Next) => {
      if (n.id === "review-changes") {
        model.markReviewed();
        setSheet("changes");
      } else if (n.id === "compare-departures") {
        // bring the timeline in, put the keyboard on its periods (arrows + Space pick), and mark it for a moment
        const el = timelineRef.current;
        el?.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
        el?.querySelector<HTMLElement>("[role='listbox']")?.focus({ preventScroll: true });
        setInvite(true);
        window.setTimeout(() => setInvite(false), 1600);
      } else if (n.id === "recheck") void model.recheckNow();
      else if (n.id === "edit-assumptions" || n.id === "set-date") setSheet("edit");
      else if (n.id === "record-outcome" || n.id === "edit-outcome") setSheet("outcome");
      else if (n.id === "plan") model.plan();
      else if (n.id === "reopen") model.reopen();
    },
    [model, reduce],
  );

  if (!trip) return null;
  const done = trip.state === "completed";
  const off = trip.state === "cancelled";
  // what the device knows per source (D's knownView): a check that failed never blanks or replaces a known forecast
  const known = model.known;
  const forecast = known.forecast?.value ?? null;
  const olderForecast = known.forecast && !known.forecast.latest ? known.forecast : null;

  const heading = <TripHeading trip={trip} port={port?.name ?? null} now={now} onRename={model.rename} id="trip-h" />;
  const feed = (
    <section className={s.since} data-testid="trip-since">
      <ChangeFeed
        planning={planning}
        result={recheck}
        checking={busy}
        now={nowIso}
        timeline={timeline}
        // when the one filled action is itself a recheck, the feed does not repeat it
        onRecheck={next?.id === "recheck" ? undefined : () => void model.recheckNow()}
        onEvidence={(c) => {
          setFocusKey(c.evidenceKey);
          setSheet("snapshot");
        }}
        actionVariant="quiet"
        headingId="trip-since"
      />
      {planning ? (
        <p className={s.planned}>
          <Button variant="quiet" className={s.inline} onClick={() => setSheet("snapshot")} data-testid="trip-open-snapshot">
            What Neritic showed on {day(planning.takenAt)}
          </Button>
          {recheck ? (
            <Button variant="quiet" className={s.inline} onClick={() => setSheet("changes")}>
              Recheck details
            </Button>
          ) : null}
        </p>
      ) : null}
    </section>
  );
  const action = <NextAction next={next} onAct={act} busy={busy} pinned={phone} />;

  // the official line, in the record's words, as Neritic showed it at planning; its verified state is the glyph
  const seasonK = known.evidence.find((e) => e.value.kind === "season") ?? null;
  const whaleK = known.evidence.find((e) => e.value.kind === "whale") ?? null;
  const season = seasonK?.value ?? null;
  const whale = whaleK?.value ?? null;
  // an official record not read by the latest check carries its age as the "old" mark and its time
  const olderOfficial = [seasonK, whaleK].find((k) => k && !k.latest) ?? null;
  const status = (
    <section className={s.status} aria-label="Official information, as last read" data-testid="trip-status">
      <Micro>Official</Micro>
      {season ? (
        <p className={s.statusLine} data-verified={season.verified === true || undefined}>
          <Glyph kind={season.verified === true ? "measured" : "unverified-coral"} label={season.verified === true ? "Verified by a person" : "Official record, not verified by a person"} />
          <span>{season.text}</span>
        </p>
      ) : (
        <p className={s.statusLine}>
          <Glyph kind="none" label="No record" />
          <span>No season record was saved. Unknown is not open.</span>
        </p>
      )}
      {whale ? (
        <p className={s.statusSub}>
          <Glyph kind={whale.verified ? "measured" : "unverified-coral"} /> {whale.title}
        </p>
      ) : null}
      {olderOfficial ? (
        <p className={s.aged} data-testid="trip-status-aged">
          <Glyph kind="old" label="Not from the latest check" /> As read {dayTime(olderOfficial.readAt)}
        </p>
      ) : null}
      {season?.url ? (
        <a className={s.src} href={season.url}>
          {season.source} page
        </a>
      ) : null}
    </section>
  );
  // the stub keeps the one figure; C's ledger (waterfall, break-even, the full worksheet) opens beside it
  const money = moneyOf(trip);
  const ledger = (
    <section className={s.money} aria-label="Economics" data-testid="trip-money">
      <Micro glyph={money ? "hypothetical" : "none"}>{money ? `${money.label} · planned` : "Economics"}</Micro>
      {money ? (
        <>
          <p className={s.moneyFig} data-testid="trip-money-figure">
            {money.text}
          </p>
          {money.be ? <p className={s.statusSub}>Break-even {money.be}</p> : null}
          {money.rec ? null : (
            <p className={s.statusSub}>
              Not profit.{" "}
              <Button variant="quiet" className={s.inline} onClick={() => setSheet("money")} aria-label={`More: ${NET_IS_NOT}`}>
                What it leaves out
              </Button>
            </p>
          )}
        </>
      ) : (
        <p className={s.statusSub}>Not modelled</p>
      )}
      <Button variant="quiet" className={s.inline} onClick={() => setSheet("money")} data-testid="trip-open-money">
        {money ? "Open the economics" : "Add costs"}
      </Button>
    </section>
  );
  const footnote = (
    <p className={s.local} data-testid="trip-local">
      <Glyph kind="none" /> Kept on this device only. Nothing is sent to Neritic.
    </p>
  );
  // the quiet group: two words in view, the rest a tap away (the one filled button is the next action)
  const ahead = trip.state === "draft" || trip.state === "planned";
  // the rarer transitions: behind More on a wide screen; in the sheet's flow on a phone (a menu that closes under a
  // thumb lets the tap land on what was beneath it)
  const rare = (close: () => void) => (
    <>
      {trip.state === "planned" ? (
        <Button
          variant="quiet"
          className={s.inline}
          onClick={() => {
            close();
            setSheet("outcome");
          }}
          data-testid="trip-complete"
        >
          Mark completed
        </Button>
      ) : null}
      {ahead ? (
        <Button
          variant="quiet"
          className={s.inline}
          onClick={() => {
            close();
            model.cancel();
          }}
          data-testid="trip-cancel"
        >
          Cancel trip
        </Button>
      ) : null}
      <Button variant="quiet" className={s.inline} href="/coast/trips" data-testid="trip-all">
        All trips
      </Button>
    </>
  );
  const quiet = (
    <div className={s.quietActions} data-testid="trip-actions">
      {ahead ? (
        <Button variant="quiet" className={s.inline} onClick={() => setSheet("edit")} data-testid="trip-edit">
          Edit plan
        </Button>
      ) : null}
      <Button variant="quiet" className={s.inline} href={`/coast/trips?id=${encodeURIComponent(trip.id)}&view=briefing`} data-testid="trip-open-briefing">
        Briefing
      </Button>
      {phone ? (
        rare(() => {})
      ) : (
        <Popover
          label="More for this trip"
          open={more}
          onOpenChange={setMore}
          trigger={
            <Button variant="quiet" className={s.inline} aria-haspopup="menu" data-testid="trip-more">
              More
            </Button>
          }
        >
          <div className={s.menu}>{rare(() => setMore(false))}</div>
        </Popover>
      )}
    </div>
  );

  const rest = (
    <div className={s.rest}>
      {done ? null : status}
      {done ? (
        // planned vs actual already shows the planned figure: here only the way to the planned worksheet
        <Button variant="quiet" className={s.inline} onClick={() => setSheet("money")} data-testid="trip-open-money">
          The planned economics
        </Button>
      ) : (
        ledger
      )}
    </div>
  );
  const departures = (
    <section ref={timelineRef} id="trip-departures" aria-labelledby="trip-dep-h" className={s.departures} data-invite={invite || undefined} data-testid="trip-departures">
      <div className={s.depHead}>
        <h2 id="trip-dep-h" className={s.depH}>
          Departure windows
        </h2>
        <Micro glyph="predicted">
          {forecast?.zone ? `NWS ${forecast.zone}` : "NWS"}
          {forecast?.issued ? ` · issued ${day(forecast.issued)} ${time(forecast.issued)}` : ""}
        </Micro>
      </div>
      {olderForecast ? (
        <p className={s.aged} data-testid="trip-forecast-aged">
          <Glyph kind="old" label="Not from the latest check" /> {olderForecast.snapshotId === planning?.id ? "From when you planned" : "From your check at"} {dayTime(olderForecast.readAt)} · the latest could not read NWS
        </p>
      ) : null}
      {now ? <Timeline known={known} limits={trip.limits} departDate={trip.depart?.date ?? null} now={now.toISOString()} timeZone={trip.depart?.timeZone} /> : null}
    </section>
  );

  // the lead of the main part: what changed (ahead), planned vs actual (completed), or the quiet cancelled line
  const lead = done ? (
    <section className={s.since} aria-label="Planned vs actual">
      <PlannedVsActual trip={trip} />
    </section>
  ) : off ? (
    <p className={s.cancelled} data-testid="trip-cancelled">
      Cancelled. Its plan, snapshots and numbers are kept.
    </p>
  ) : (
    feed
  );

  return (
    <>
      <TripChart trip={trip} snapshot={latest?.mode === "live" ? latest : planning} port={port} />
      <PageLayout mode="stage" paneKey="trips" chartLabel={port?.name} phoneMap={0.3} />
      <LayoutGroup>
        <Ticket key={trip.id} at={port ? { lon: port.lon, lat: port.lat } : null} anchorY={130} width={440} labelledBy="trip-h" className={s.ticket} data-testid="trip-workspace" data-state={trip.state}>
          <TicketMain className={s.main}>
            <motion.div layout={reduce ? false : "position"} transition={reflow} className={s.mainInner} data-off={off || undefined}>
              {heading}
              {lead}
              {action}
              {phone ? null : quiet}
            </motion.div>
          </TicketMain>
          {phone ? null : (
            <TicketStub className={s.stub}>
              {rest}
              {footnote}
            </TicketStub>
          )}
        </Ticket>
        {off || done ? null : (
          <Dock label="Departure windows" align="left" width={760} className={s.dock}>
            {departures}
          </Dock>
        )}
        {phone ? (
          <Dock label="This trip" align="left" className={s.dock}>
            {rest}
            {quiet}
            {footnote}
          </Dock>
        ) : null}
      </LayoutGroup>
      {model.said ? (
        <p className={s.said} role="status" data-testid="trip-said">
          {model.said}
        </p>
      ) : null}

      <Drawer title="Since you planned this trip" description="Changed, the same, couldn't check, and where to verify directly." open={sheet === "changes"} onOpenChange={(o) => setSheet(o ? "changes" : null)}>
        <RecheckPanel recheck={recheck} checking={busy} onRecheck={() => void model.recheckNow()} now={nowIso} actionVariant="secondary" savedFrom={model.savedFrom} />
      </Drawer>
      <Drawer title="What Neritic showed you" description="The planning snapshot, as it was. It is never overwritten." open={sheet === "snapshot"} onOpenChange={(o) => setSheet(o ? "snapshot" : null)}>
        {planning ? <SnapshotView snapshot={planning} now={nowIso} superseded={superseded} focusKey={focusKey} /> : null}
        {planning && trip.state !== "completed" && trip.state !== "cancelled" ? (
          <Button variant="quiet" onClick={() => void model.refreshPlanning()} busy={busy} busyLabel="Reading the sources" data-testid="trip-refresh-planning">
            Refresh the planning snapshot
          </Button>
        ) : null}
      </Drawer>
      <Drawer
        title="Economics"
        description="Your numbers, through the same worksheet as My Coast. Hypothetical: nothing is filled in or forecast."
        open={sheet === "money"}
        onOpenChange={(o) => setSheet(o ? "money" : null)}
      >
        <Ledger trip={trip} now={now} readOnly={done || off} onSave={(economics, refs) => model.edit({ economics, refs })} />
      </Drawer>
      <Drawer
        title="Edit plan"
        description="Your date, your own limits and the area. Limits are your own, to read the forecast against: not a rule or a standard."
        open={sheet === "edit"}
        onOpenChange={(o) => setSheet(o ? "edit" : null)}
      >
        {sheet === "edit" ? <EditPlan model={model} onDone={() => setSheet(null)} /> : null}
      </Drawer>
      <OutcomeSheet
        open={sheet === "outcome"}
        onOpenChange={(o) => setSheet(o ? "outcome" : null)}
        trip={trip}
        onSave={(o) => {
          if (model.recordOutcome(o)) setSheet(null);
        }}
      />
    </>
  );
}

/** The plan's date, time, limits and area, edited with the planner's own fields. */
function EditPlan({ model, onDone }: { model: TripModel; onDone: () => void }) {
  const t = model.trip!;
  const [p, setP] = useState<PlanDraft>(() => ({
    ...EMPTY_PLAN,
    date: t.depart?.date ?? "",
    time: t.depart?.time ?? "",
    windKt: t.limits.windKt?.toString() ?? "",
    seasFt: t.limits.seasFt?.toString() ?? "",
    area: t.area ?? "",
  }));
  const set = useCallback((patch: Partial<PlanDraft>) => setP((x) => ({ ...x, ...patch })), []);
  return (
    <form
      className={s.edit}
      onSubmit={(e) => {
        e.preventDefault();
        // a draft that now has its date is planned in the same step
        if (
          model.edit(
            {
              depart: departure(p.date, p.time, t.depart?.timeZone),
              limits: {
                windKt: limitValue(p.windKt),
                seasFt: limitValue(p.seasFt),
              },
              area: p.area,
            },
            { plan: true },
          )
        )
          onDone();
      }}
    >
      <PlanFields plan={p} set={set} />
      <Button variant="secondary" type="submit" data-testid="trip-edit-save">
        Save plan
      </Button>
    </form>
  );
}

/** The departure timeline (B) over the newest live snapshot; it compares the two periods the reader picks itself. */
function Timeline({ known, limits, departDate, now, timeZone }: { known: KnownView; limits: OperatingLimits; departDate: string | null; now: string; timeZone?: string }) {
  const alerts = useMemo<AdvisoryIn[]>(
    () =>
      known.evidence
        .map((k) => k.value)
        .filter((e) => e.kind === "advisory")
        .map((e) => ({ event: e.title, headline: e.text, url: e.url ?? "", onset: (e.fields.onset as string | null) ?? null, ends: (e.fields.ends as string | null) ?? null })),
    [known],
  );
  return <DepartureTimeline forecast={known.forecast?.value ?? null} tides={known.tides?.value.events ?? null} alerts={alerts} limits={limits} departDate={departDate} now={now} timeZone={timeZone} />;
}

/** The trip's one money figure through the engine of record: net to boat (commercial) or trip cost (recreational). */
const NET_IS_NOT = "Net to boat comes before insurance, permits, maintenance, gear loss, loan payments and the skipper's own pay. It is not profit or take-home pay.";

function moneyOf(t: Trip): { label: string; text: string; be: string | null; rec: boolean } | null {
  if (!t.economics) return null;
  const f = tripFishery(t.fishery);
  const r = evaluateWorkspace(fromDraft(t.economics, t.fishery), f);
  const rec = r.audience === "recreational";
  const v = rec ? r.out.operating : r.out.net;
  const be = r.out.breakEvenCatch;
  return {
    label: rec ? "Trip cost" : "Net to boat",
    text: v.status === "value" ? fmtMoney(v.value) : "$ —",
    be: !rec && be.status === "value" ? breakEvenCatchText(be.value, f?.unit ?? "lb") : null,
    rec,
  };
}
