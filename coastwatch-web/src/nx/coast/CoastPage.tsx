"use client";

import { MotionConfig } from "motion/react";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { setChartHighlight } from "@/nx/chart/highlight";
import { ChartRegulation, rampZoneAt } from "@/nx/chart/regulation";
import { ChartScene, tallBoundsAround, type ChartScene as Scene } from "@/nx/chart/scene";
import { Button, ChartToggle, Desk, DeskAside, DeskBody, DeskFoot, DeskHeader, DeskMain, Drawer, Legend, PageLayout } from "@/nx/ds";
import { openPortFinder } from "@/nx/shell/finder";
import { savePort, useCurrentPort } from "@/nx/shell/port";
import { portScene, zoneScene } from "@/nx/shell/views";
import type { FisheryId } from "@/pilot/contracts";
import { carriedWords, readCarried, tripFlag } from "./context";
import { audienceOf, COAST_FISHERIES, RECREATIONAL_FISHERIES, tripFishery } from "./fisheries";
import { Gates } from "./Gates";
import { capacity } from "./scenarios";
import { SeaWindow } from "./SeaWindow";
import { TripWorkspace } from "./TripWorkspace";
import { useCoastView, useFishery, useGates, useSea } from "./useCoast";
import { useTripData } from "./useTripStore";
import s from "./coast.module.css";
import t from "./trip.module.css";

/**
 * My Coast (journeys.md, "Planning a trip"): a desk page (v13 README §1) in two states on one address. The
 * brief is the gates for the chosen fishery and the sea window (the reader chooses the day). The trip
 * (`?view=trip`) is the money workspace from the reader's own numbers. The header keeps the port, the fishery,
 * whether the trip is commercial or recreational, "hypothetical" and the official season status in view in
 * both, however far the page is scrolled. Nothing on it is a verdict on a trip (T36, T46).
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
  const sea = useSea(view === "brief" ? (port?.key ?? null) : null);
  const saved = capacity(useTripData()).used;
  // the clock for reference ages (the diesel week): the server's, as the gates and sea answers carry it (a
  // fixture preview pins it), else the device's after hydration
  const [deviceNow, setDeviceNow] = useState<Date | null>(null);
  useEffect(() => setDeviceNow(new Date()), []);
  const served = gates.state === "ok" ? gates.data.now : sea.state === "ok" ? sea.data.now : null;
  const now = useMemo(() => (served ? new Date(served) : deviceNow), [served, deviceNow]);
  const selectId = useId();
  const onFishery = useCallback((id: string) => tripFishery(id) && setFishery(id as FisheryId), [setFishery]);

  // what the reader brought from Explore: said in the header, ringed on the chart, never used by the worksheet
  const carried = useMemo(() => readCarried(sp), [sp]);
  const brought = carriedWords(carried);
  const at = carried.at;
  useEffect(() => {
    if (!at) return;
    setChartHighlight({ lon: at.lon, lat: at.lat, label: "From Explore" });
    return () => setChartHighlight(null);
  }, [at]);

  // the official status, kept in the header: the record's own sentence, whose it is, and that nobody verified it
  const season = gates.state === "ok" ? gates.data.season : null;
  const flag = tripFlag(season, fishery);
  const [gatesOpen, setGatesOpen] = useState(false);
  const chip = !port
    ? null
    : !fishery
      ? { words: "Choose a fishery to see its season status.", tone: "ink" }
      : gates.state === "loading" || gates.state === "idle"
        ? { words: "Reading the season status…", tone: "ink" }
        : gates.state === "error"
          ? { words: "The season status could not be read. Unknown is not open.", tone: "regulation" }
          : season
            ? { words: `${season.text} ${season.source}${season.verified === true ? "" : " · not verified"}`, tone: season.tone }
            : { words: "No season record loaded. Unknown is not open.", tone: "regulation" };

  // the sticky header's height, for what sticks under it (the trip's result)
  const page = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = page.current;
    const head = el?.querySelector<HTMLElement>("[data-nx-desk-header]");
    if (!el || !head || typeof ResizeObserver === "undefined") return;
    const set = () => {
      el.style.setProperty("--trip-top", `${head.offsetHeight}px`);
    };
    set();
    const ro = new ResizeObserver(set);
    ro.observe(head);
    return () => ro.disconnect();
    // each state renders its own header element
  }, [view]);

  // a crab trip's water is its RAMP zone, drawn whole (already a tall box, as the pane beside the desk is);
  // otherwise the port's own water, with a run of coast either side of it for a tall pane
  const scene = useMemo((): Scene => {
    const zone = fishery?.ramp && port ? zoneScene("coast", port, rampZoneAt(port.lat)) : null;
    if (zone) return zone;
    const around = portScene("coast", port, 30);
    return port && "bounds" in around.view ? { ...around, view: { ...around.view, tall: tallBoundsAround([port.lon - 0.12, port.lat], 55, 22) } } : around;
  }, [fishery, port]);

  const trip = view === "trip";
  // short enough to stay whole on a 320 px screen once the header condenses: the port is in the title beside it
  // and the fishery in the control under it, so all of them stay in view together
  // The way back to the gates rides on the same line in the trip, so the fishery below has the header's whole
  // width (its name is never cut) and the condensed header stays three short lines on a 320 px screen, where
  // the word "trip" gives way (the title beside it says "A trip from …").
  const kind = fishery ? (audience === "recreational" ? "Recreational, cost only" : "Commercial") : null;
  const context = (
    <span className={t.ctx}>
      <span>
        {trip ? (
          <>
            Hypothetical<span className={t.ctxWord}> trip</span>
          </>
        ) : (
          "My Coast"
        )}
        {kind ? ` · ${kind}` : ""}
      </span>
      {trip ? (
        <Button variant="quiet" className={t.ctxLink} onClick={() => setView("brief")} aria-label="Back to the gates and sea window" data-testid="coast-to-brief">
          Gates and sea
        </Button>
      ) : null}
    </span>
  );
  /**
   * The desk header, the same in both states: context, title, and pinned under it the fishery, the official season
   * line (it opens the gates) and anything carried from Explore. The trip hands it its saving controls as actions;
   * the brief leads with the way into the trip.
   */
  const header = (actions: ReactNode) => (
    <DeskHeader
      context={context}
      title={port ? `A trip from ${port.name}` : "Plan a trip"}
      titleProps={{ "data-testid": "coast-title" }}
      status={trip ? undefined : "The gates for your fishery and the sea window. Then work the money out from your own numbers."}
      actions={
        <>
          {actions}
          <ChartToggle />
        </>
      }
      pinned={
        <div className={t.pinned}>
          <div className={s.fishery}>
            <label htmlFor={selectId} className={s.fisheryLabel}>
              Fishery
            </label>
            <select id={selectId} className={s.select} value={fishery?.id ?? ""} onChange={(e) => e.target.value && setFishery(e.target.value as FisheryId)} data-testid="coast-fishery">
              {fishery ? null : <option value="">Choose one</option>}
              <optgroup label="Commercial">
                {COAST_FISHERIES.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.label}
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
          </div>
          {chip ? (
            <Button variant="quiet" className={t.chip} data-tone={chip.tone} onClick={() => setGatesOpen(true)} aria-haspopup="dialog" data-testid="coast-status">
              {chip.words}
            </Button>
          ) : null}
          {brought.point || brought.date ? (
            <p className={t.carried} data-testid="coast-carried">
              {[brought.point, brought.date].filter(Boolean).join(" · ")}
            </p>
          ) : null}
        </div>
      }
    >
      {trip ? null : (
        <div className={t.lead}>
          <Button variant="primary" onClick={() => setView("trip")} data-testid="coast-to-trip">
            Work out the money
          </Button>
        </div>
      )}
    </DeskHeader>
  );

  return (
    <MotionConfig reducedMotion="user">
      <ChartScene scene={scene} />
      {/* the chosen fishery's regulation on the chart: RAMP's zone lines for crab, the port's zone heavier (E T43) */}
      <ChartRegulation spec={fishery?.ramp ? { zones: { highlight: port ? rampZoneAt(port.lat) : null } } : null} />
      <PageLayout mode="desk" paneKey="coast" chartLabel={port?.name} />
      <article className={s.page} data-testid="nx-coast" data-view={view} ref={page}>
        <Desk>
          {trip ? null : header(null)}

          {trip ? (
            <TripWorkspace fishery={fishery} placeKey={port?.key ?? ""} onFishery={onFishery} now={now} flag={flag} header={header} />
          ) : (
            <DeskBody layout="main-aside" aside="start">
              <DeskAside label="Gates">
                <section className={s.section} aria-labelledby="coast-gates">
                  <h2 id="coast-gates" className={s.h2}>
                    Gates
                  </h2>
                  {!port ? (
                    <div className={s.empty}>
                      <p>Choose your port to see its season status, whale measures and latest test.</p>
                      <Button variant="primary" onClick={openPortFinder}>
                        Find your port
                      </Button>
                    </div>
                  ) : !fishery ? (
                    <p className={s.empty}>Choose a fishery above to see its season status, whale measures and latest test.</p>
                  ) : (
                    <Gates gates={gates} fishery={fishery} />
                  )}
                </section>
              </DeskAside>
              <DeskMain>
                {port ? (
                  <section className={s.section} aria-labelledby="coast-sea">
                    <h2 id="coast-sea" className={s.h2}>
                      Sea window
                    </h2>
                    <p className={s.sub}>Choose the day you are weighing.</p>
                    <SeaWindow sea={sea} placeKey={port.key} />
                  </section>
                ) : null}
                <section className={t.briefMoney} aria-labelledby="coast-money">
                  <h2 id="coast-money" className={s.h2}>
                    The money
                  </h2>
                  <p>
                    {audience === "recreational"
                      ? "What the trip would cost, split between the people aboard, from your own numbers."
                      : "Gross, costs, crew pay, net to boat and the catch and price that break even, from your own numbers."}{" "}
                    Hypothetical: nothing is filled in or forecast.
                  </p>
                  <Button variant="secondary" onClick={() => setView("trip")} data-testid="coast-to-trip-2">
                    {saved ? `Open the trip workspace (${saved} saved)` : "Open the trip workspace"}
                  </Button>
                </section>
              </DeskMain>
            </DeskBody>
          )}

          <DeskFoot>
            <Legend kinds={["measured", "model", "official"]} words={{ model: "Forecast or prediction", official: "Official record, not verified" }} title="Kinds on this page" />
            <p className={s.small}>Season and whale measures are CDFW&apos;s; tests are CDPH&apos;s; the forecast is NWS&apos;s. Neritic links to them and does not replace them. Not for navigation.</p>
          </DeskFoot>
        </Desk>

        <Drawer title={fishery ? `Gates for ${fishery.label}` : "Gates"} description="Season status, whale measures and the latest test, each with its source and date." open={gatesOpen} onOpenChange={setGatesOpen}>
          {fishery && port ? <Gates gates={gates} fishery={fishery} /> : <p className={s.empty}>Choose a port and a fishery to see its season status, whale measures and latest test.</p>}
        </Drawer>
      </article>
    </MotionConfig>
  );
}
