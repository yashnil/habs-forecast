"use client";

import { MotionConfig } from "motion/react";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { setChartHighlight } from "@/nx/chart/highlight";
import { Button, Dock, Drawer, Glyph, Micro, PageLayout, Ticket, TicketMain, TicketStub } from "@/nx/ds";
import { openPortFinder } from "@/nx/shell/finder";
import { savePort, useCurrentPort } from "@/nx/shell/port";
import type { FisheryId } from "@/pilot/contracts";
import { CoastMap } from "./CoastMap";
import { carriedWords, readCarried } from "./context";
import { audienceOf, COAST_FISHERIES, RECREATIONAL_FISHERIES, tripFishery, type CoastFishery } from "./fisheries";
import { Gates } from "./Gates";
import { seasonWarning } from "./pass";
import { capacity } from "./scenarios";
import { SeaWindow } from "./SeaWindow";
import { TripWorkspace } from "./TripWorkspace";
import { useCoastView, useFishery, useGates, useSea } from "./useCoast";
import { useTripData } from "./useTripStore";
import s from "./coast.module.css";
import t from "./trip.module.css";

/**
 * My Coast (v14 synthesis): the chart is the page. Two states on one address. The brief is a ticket beside the port
 * (the fishery, its official season line, the gates) with the sea window in a dock along the chart's foot and the way
 * into the trip. The trip (`?view=trip`) is the perforated pass along the foot of the chart (./TripWorkspace). Both
 * keep the port, the fishery, "hypothetical" and the official season status in view, and nothing on either is a
 * verdict on a trip (T36, T46). The chart draws the port, its whale zone, the depth, the buoy and the last CDPH test.
 */
export function CoastPage() {
  const { port, key, fromUrl } = useCurrentPort();
  useEffect(() => {
    if (fromUrl && key) savePort(key);
  }, [key, fromUrl]);
  const sp = useSearchParams();
  const [fisheryId, setFishery] = useFishery();
  const fishery = tripFishery(fisheryId);
  const audience = audienceOf(fishery);
  const [view, setView] = useCoastView();
  const gates = useGates(port?.key ?? null, fishery?.id ?? null);
  const sea = useSea(port?.key ?? null);
  const saved = capacity(useTripData()).used;
  // the clock for reference ages (the diesel week): the server's, as the gates and sea answers carry it (a
  // fixture preview pins it), else the device's after hydration
  const [deviceNow, setDeviceNow] = useState<Date | null>(null);
  useEffect(() => setDeviceNow(new Date()), []);
  const served = gates.state === "ok" ? gates.data.now : sea.state === "ok" ? sea.data.now : null;
  const now = useMemo(() => (served ? new Date(served) : deviceNow), [served, deviceNow]);
  const selectId = useId();
  const onFishery = useCallback((id: string) => tripFishery(id) && setFishery(id as FisheryId), [setFishery]);

  // what the reader brought from Explore: said on the page, ringed on the chart, never used by the worksheet
  const carried = useMemo(() => readCarried(sp), [sp]);
  const brought = carriedWords(carried);
  const at = carried.at;
  useEffect(() => {
    if (!at) return;
    setChartHighlight({ lon: at.lon, lat: at.lat, label: "From Explore" });
    return () => setChartHighlight(null);
  }, [at]);

  // the official status: the record's own sentence, whose it is, and that nobody verified it
  const season = gates.state === "ok" ? gates.data.season : null;
  const [gatesOpen, setGatesOpen] = useState(false);
  const chip = !port
    ? null
    : !fishery
      ? { words: "Choose a fishery to see its season status.", tone: "ink", src: null }
      : gates.state === "loading" || gates.state === "idle"
        ? { words: "Reading the season status…", tone: "ink", src: null }
        : gates.state === "error"
          ? { words: "The season status could not be read. Unknown is not open.", tone: "regulation", src: null }
          : season
            ? { words: season.text, tone: season.tone, src: `${season.source}${season.verified === true ? "" : " · not verified"}` }
            : { words: "No season record loaded. Unknown is not open.", tone: "regulation", src: null };

  const trip = view === "trip";
  const kind = fishery ? (audience === "recreational" ? "Recreational, cost only" : "Commercial") : null;
  const carriedLine =
    brought.point || brought.date ? (
      <p className={t.carried} data-testid="coast-carried">
        {[brought.point, brought.date].filter(Boolean).join(" · ")}
      </p>
    ) : null;
  const select = (className: string, labelled: boolean) => (
    <select
      id={selectId}
      className={className}
      value={fishery?.id ?? ""}
      onChange={(e) => e.target.value && setFishery(e.target.value as FisheryId)}
      aria-label={labelled ? undefined : "Fishery"}
      data-testid="coast-fishery"
    >
      {fishery ? null : <option value="">Choose a fishery</option>}
      <optgroup label="Commercial">
        {COAST_FISHERIES.map((f) => (
          <option key={f.id} value={f.id}>
            {trip ? `${f.label}, commercial` : f.label}
          </option>
        ))}
      </optgroup>
      <optgroup label="Recreational (cost only)">
        {RECREATIONAL_FISHERIES.map((f) => (
          <option key={f.id} value={f.id}>
            {f.label}
          </option>
        ))}
      </optgroup>
    </select>
  );

  // the trip's strip: hypothetical, the port and fishery, and the season line in the record's words (it opens the gates)
  const warning = seasonWarning(season);
  const strip = (
    <header className={t.strip} data-nx-desk-header="">
      <div className={t.hy}>
        <Micro glyph="hypothetical" className={t.hyMicro} aria-hidden="true">
          Hypothetical trip{audience === "recreational" ? " · cost only" : ""}
        </Micro>
        <span className={t.vh}>Hypothetical trip{audience === "recreational" ? ", cost only" : ""}: every figure is from your numbers. </span>
        <h1 className={t.where} data-testid="coast-title">
          <span className={t.vh}>A trip from </span>
          {port ? port.name : "No port chosen"}
        </h1>
        <span className={t.dot} aria-hidden="true">
          ·
        </span>
        {select(t.select, false)}
      </div>
      {chip ? (
        <Button variant="quiet" className={t.warn} data-tone={warning ? "regulation" : chip.tone} onClick={() => setGatesOpen(true)} aria-haspopup="dialog" title={chip.src ? `${chip.words} ${chip.src}` : undefined} data-testid="coast-status">
          {warning || chip.tone !== "ink" ? <Glyph kind="unverified-coral" className={t.warnG} /> : null}
          <span>{warning ?? chip.words}</span>
          {/* the ring is the grammar's "official record, not verified"; its words for a screen reader, the record a tap away */}
          {chip.src ? <span className={t.vh}> {chip.src}</span> : null}
        </Button>
      ) : null}
      {carriedLine}
    </header>
  );

  return (
    <MotionConfig reducedMotion="user">
      <CoastMap port={port} fishery={fishery} gates={gates} sea={sea} view={view} />
      <PageLayout mode="stage" paneKey="coast" chartLabel={port?.name} phoneMap={trip ? 0.42 : 0.4} />
      <article className={s.page} data-testid="nx-coast" data-view={view}>
        {trip ? (
          <TripWorkspace fishery={fishery} placeKey={port?.key ?? ""} at={port ? { lon: port.lon, lat: port.lat } : null} onFishery={onFishery} now={now} strip={strip} toBrief={() => setView("brief")} />
        ) : (
          <CoastBrief
            port={port}
            fishery={fishery}
            kind={kind}
            chip={chip}
            select={select(s.select, true)}
            selectId={selectId}
            carriedLine={carriedLine}
            gates={gates}
            sea={sea}
            saved={saved}
            onGates={() => setGatesOpen(true)}
            onTrip={() => setView("trip")}
          />
        )}

        <Drawer title={fishery ? `Gates for ${fishery.label}` : "Gates"} description="Season status, whale measures and the latest test, each with its source and date." open={gatesOpen} onOpenChange={setGatesOpen}>
          {fishery && port ? <Gates gates={gates} fishery={fishery} /> : <p className={s.empty}>Choose a port and a fishery to see its season status, whale measures and latest test.</p>}
        </Drawer>
      </article>
    </MotionConfig>
  );
}

type Chip = { words: string; tone: string; src: string | null } | null;

/**
 * The brief (v14): a ticket beside the port (the fishery, the official season line, the gates, and in its stub the way
 * into the trip) and the sea window in a dock along the chart's foot, the reader choosing the day.
 */
function CoastBrief({
  port,
  fishery,
  kind,
  chip,
  select,
  selectId,
  carriedLine,
  gates,
  sea,
  saved,
  onGates,
  onTrip,
}: {
  port: ReturnType<typeof useCurrentPort>["port"];
  fishery: CoastFishery | null;
  kind: string | null;
  chip: Chip;
  select: React.ReactNode;
  selectId: string;
  carriedLine: React.ReactNode;
  gates: ReturnType<typeof useGates>;
  sea: ReturnType<typeof useSea>;
  saved: number;
  onGates: () => void;
  onTrip: () => void;
}) {
  const rec = fishery && audienceOf(fishery) === "recreational";
  return (
    <>
      <Ticket key={port?.key ?? "none"} at={port ? { lon: port.lon, lat: port.lat } : null} anchorY={120} width={404} labelledBy="coast-title" className={s.ticket}>
        <TicketMain className={s.ticketMain}>
          <div className={s.head} data-nx-desk-header="">
            <Micro>{kind ? `My Coast · ${kind}` : "My Coast"}</Micro>
            <h1 id="coast-title" className={s.title} data-testid="coast-title">
              {port ? `A trip from ${port.name}` : "Plan a trip"}
            </h1>
            <div className={s.fishery}>
              <label htmlFor={selectId} className={s.fisheryLabel}>
                Fishery
              </label>
              {select}
            </div>
            {chip ? (
              <Button variant="quiet" className={s.status} data-tone={chip.tone} onClick={onGates} aria-haspopup="dialog" data-testid="coast-status">
                {chip.tone !== "ink" ? <Glyph kind="unverified-coral" className={s.statusG} /> : null}
                <span>
                  {chip.words}
                  {chip.src ? <span className={s.statusSrc}> {chip.src}</span> : null}
                </span>
              </Button>
            ) : null}
            {carriedLine}
          </div>
          <section className={s.section} aria-labelledby="coast-gates">
            <h2 id="coast-gates" className={s.h2}>
              Gates
            </h2>
            {!port ? (
              <div className={s.empty}>
                <p>Choose your port to see its season status, whale measures and latest test.</p>
                <Button variant="secondary" onClick={openPortFinder}>
                  Find your port
                </Button>
              </div>
            ) : !fishery ? (
              <p className={s.empty}>Choose a fishery above to see its season status, whale measures and latest test.</p>
            ) : (
              <Gates gates={gates} fishery={fishery} />
            )}
          </section>
        </TicketMain>
        <TicketStub className={s.ticketStub}>
          <section aria-labelledby="coast-money">
            <h2 id="coast-money" className={s.h2}>
              The money
            </h2>
            <p className={s.sub}>
              {rec ? "What the trip would cost, split between the people aboard, from your own numbers." : "Gross, costs, crew pay, net to boat and the catch and price that break even, from your own numbers."} Hypothetical: nothing is filled in or forecast.
            </p>
            <Button variant="primary" full onClick={onTrip} data-testid="coast-to-trip">
              Work out the money
            </Button>
            <Button variant="quiet" className={s.toTrip} onClick={onTrip} data-testid="coast-to-trip-2">
              {saved ? `Open the trip workspace (${saved} saved)` : "Open the trip workspace"}
            </Button>
          </section>
        </TicketStub>
      </Ticket>
      {port ? (
        <Dock label="Sea window" align="left" width={560} className={s.dock}>
          <section aria-labelledby="coast-sea" id="coast-sea-wrap">
            <h2 id="coast-sea" className={s.h2}>
              Sea window
            </h2>
            <p className={s.sub}>Choose the day you are weighing.</p>
            <SeaWindow sea={sea} placeKey={port.key} />
            <p className={s.small}>Season and whale measures are CDFW&apos;s; tests are CDPH&apos;s; the forecast is NWS&apos;s. Neritic links to them and does not replace them. Not for navigation.</p>
          </section>
        </Dock>
      ) : null}
    </>
  );
}
