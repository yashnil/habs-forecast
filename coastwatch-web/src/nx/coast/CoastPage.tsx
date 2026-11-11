"use client";

import { MotionConfig } from "motion/react";
import { useEffect, useId, useState } from "react";
import { ChartScene } from "@/nx/chart/scene";
import { Legend } from "@/nx/ds";
import { openPortFinder } from "@/nx/shell/PortFinder";
import { savePort, useCurrentPort } from "@/nx/shell/port";
import { portScene } from "@/nx/shell/views";
import type { FisheryId } from "@/pilot/contracts";
import { COAST_FISHERIES, coastFishery } from "./fisheries";
import { Gates } from "./Gates";
import { SeaWindow } from "./SeaWindow";
import { useFishery, useGates, useSea } from "./useCoast";
import { Worksheet } from "./Worksheet";
import s from "./coast.module.css";

/**
 * My Coast (journeys.md, "Planning a trip"): one page that reads top to bottom in the margin while the chart
 * shows the trip's water. The gates for the chosen fishery, then the sea window (the reader chooses the day),
 * then the worksheet from the reader's own numbers. Nothing on it is a verdict on a trip (T36, T46).
 */
export function CoastPage() {
  const { port, key, fromUrl } = useCurrentPort();
  useEffect(() => {
    if (fromUrl && key) savePort(key);
  }, [key, fromUrl]);
  const [fisheryId, setFishery] = useFishery();
  const fishery = coastFishery(fisheryId);
  const gates = useGates(port?.key ?? null, fishery?.id ?? null);
  const sea = useSea(port?.key ?? null);
  // the clock for reference ages (the diesel week): the device's, after hydration
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => setNow(new Date()), []);
  const selectId = useId();

  return (
    <MotionConfig reducedMotion="user">
      <ChartScene scene={portScene("coast", port, 30)} />
      <article className={s.page} data-testid="nx-coast">
        <header className={s.head}>
          <p className={`n12-kicker ${s.kicker}`}>My Coast</p>
          <h1 className={`n12-statement n12-statement-m ${s.title}`}>{port ? `A trip from ${port.name}` : "Plan a trip"}</h1>
          <p className={s.lead}>The gates for your fishery, the sea window, then a worksheet from your own numbers.</p>
          <div className={s.fishery}>
            <label htmlFor={selectId} className={s.fisheryLabel}>
              Fishery
            </label>
            <select id={selectId} className={s.select} value={fishery?.id ?? ""} onChange={(e) => e.target.value && setFishery(e.target.value as FisheryId)} data-testid="coast-fishery">
              {fishery ? null : <option value="">Choose one</option>}
              {COAST_FISHERIES.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>
        </header>

        <section className={s.section} aria-labelledby="coast-gates">
          <h2 id="coast-gates" className={s.h2}>
            Gates
          </h2>
          {!port ? (
            <div className={s.empty}>
              <p>Choose your port to see its season status, whale measures and latest test.</p>
              <button type="button" className="n12-primary" onClick={openPortFinder}>
                Find your port
              </button>
            </div>
          ) : !fishery ? (
            <p className={s.empty}>Choose a fishery above to see its season status, whale measures and latest test.</p>
          ) : (
            <Gates gates={gates} fishery={fishery} />
          )}
        </section>

        {port ? (
          <section className={s.section} aria-labelledby="coast-sea">
            <h2 id="coast-sea" className={s.h2}>
              Sea window
            </h2>
            <p className={s.sub}>Choose the day you are weighing.</p>
            <SeaWindow sea={sea} placeKey={port.key} />
          </section>
        ) : null}

        <section className={s.section} aria-labelledby="coast-sheet">
          <h2 id="coast-sheet" className={s.h2}>
            Worksheet
          </h2>
          <p className={s.sub}>Your own numbers; nothing is filled in.</p>
          <Worksheet fishery={fishery} placeKey={port?.key ?? ""} onFishery={(id) => coastFishery(id) && setFishery(id as FisheryId)} now={now} />
        </section>

        <footer className={s.foot}>
          <Legend kinds={["measured", "model", "official"]} words={{ model: "Forecast or prediction", official: "Official record, not verified" }} title="Kinds on this page" />
          <p className={s.small}>Season and whale measures are CDFW&apos;s; tests are CDPH&apos;s; the forecast is NWS&apos;s. Neritic links to them and does not replace them. Not for navigation.</p>
        </footer>
      </article>
    </MotionConfig>
  );
}
