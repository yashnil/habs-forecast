"use client";

import Link from "next/link";
import { useEffect, useLayoutEffect, useSyncExternalStore } from "react";
import { boundsAround, ChartScene, type ChartScene as Scene } from "@/nx/chart/scene";
import { STILL } from "@/nx/chart/style";
import type { Bounds } from "@/nx/chart/mercator";
import { SkipOpening } from "@/nx/opening/Opening";
import { PlaceholderMargin } from "./Margin";
import { openPortFinder } from "./PortFinder";
import { savePort, useCurrentPort, usePorts, useSavedPortKey, type ShellPort } from "./port";
import { withPlace } from "./Shell";

/** California's coast, Oregon to Mexico: the view with no port chosen. */
export const CA_COAST: Bounds = [
  [-124.9, 32.4],
  [-117.0, 42.0],
];

export const HOME_SCENE: Scene = { id: "home", view: { cover: STILL.bounds, position: [0.5, 0.5] }, interactive: false };

const focusOf = (p: ShellPort | null) => (p ? { key: p.key, name: p.name, lon: p.lon, lat: p.lat } : null);

/** A destination's scene around the current port, or the whole coast. `km` is the half-width around the port. */
export function portScene(page: string, port: ShellPort | null, km: number): Scene {
  if (!port) return { id: `${page}:coast`, view: { bounds: CA_COAST } };
  // California's water is to the west: the view leans seaward, so the port sits right of centre with its sea before it
  const lean = (km * 0.45) / (111.32 * Math.cos((port.lat * Math.PI) / 180));
  return { id: `${page}:${port.key}`, view: { bounds: boundsAround([port.lon - lean, port.lat], km, km * 0.8) }, focus: focusOf(port) };
}

/** Opening a destination at a port (a deep link or a tab) makes it the reader's port on this device. */
function useRememberPort(key: string | null, fromUrl: boolean) {
  useEffect(() => {
    if (fromUrl && key) savePort(key);
  }, [key, fromUrl]);
}

/** Home: the statement and Find your port (journeys.md Journey 1). The opening plays behind it once. */
export function HomeMargin() {
  return (
    <section className="nx-home" data-nx-home data-testid="nx-home">
      <p className="n12-kicker">Free, for California commercial fishermen and coastal researchers</p>
      <h1 className="n12-statement n12-statement-l">Season status, crab tests, whale measures and the next decision date for your port.</h1>
      <div className="n12-actions">
        <button type="button" className="n12-primary" onClick={openPortFinder} data-testid="nx-find-port">
          Find your port
        </button>
        <SkipOpening />
      </div>
      <p className="n12-quiet-links">
        <Link href="/explore">See the water</Link>
        <Link href="/coast">Plan a trip</Link>
        <Link href="/research">Check the record</Link>
      </p>
      <p className="n12-small">Official notices come from CDFW and CDPH; Neritic links to them and does not replace them.</p>
    </section>
  );
}

/**
 * At `/` the server cannot know the port, so the name is written by a few bytes of script right after the
 * heading, before it paints (the pre-paint entry put the port key on <html>). Hydration keeps that text.
 */
function RootPortName({ name }: { name: string | null }) {
  const ports = usePorts();
  const names = Object.fromEntries(ports.map((p) => [p.key, p.name]));
  const js = `(function(){try{var k=document.documentElement.getAttribute("data-nx-port"),n=${JSON.stringify(names).replace(/</g, "\\u003c")}[k],s=document.currentScript;if(n&&s&&s.previousElementSibling)s.previousElementSibling.textContent=n}catch(e){}})()`;
  return (
    <>
      <span suppressHydrationWarning>{name ?? ""}</span>
      <script suppressHydrationWarning dangerouslySetInnerHTML={{ __html: js }} />
    </>
  );
}

/** Today's margin until the Today feature lands: the port, and what this page is for. */
export function TodayMargin({ atRoot = false }: { atRoot?: boolean }) {
  const { port, key, fromUrl, pending } = useCurrentPort();
  useRememberPort(key, fromUrl);
  const known = !!port || (atRoot && pending);
  return (
    <>
      {!atRoot ? <ChartScene scene={portScene("today", port, 14)} /> : null}
      <PlaceholderMargin
        testid="nx-today"
        kicker={known ? "Today" : undefined}
        title={atRoot ? <RootPortName name={port?.name ?? null} /> : port ? port.name : pending ? "Today" : "Today at your port"}
        sentence={known ? "What changed at your port, what needs attention, and the next date that matters." : "Choose your port to see its season status, the latest tests and the next decision date."}
      >
        {!known && !pending ? (
          <div className="n12-actions">
            <button type="button" className="n12-primary" onClick={openPortFinder}>
              Find your port
            </button>
          </div>
        ) : null}
      </PlaceholderMargin>
    </>
  );
}

export function ExploreMargin() {
  const { port, key, fromUrl } = useCurrentPort();
  useRememberPort(key, fromUrl);
  return (
    <>
      <ChartScene scene={portScene("explore", port, 70)} />
      <PlaceholderMargin testid="nx-explore" kicker="Explore" title="The water, as last measured" sentence="Observations and models in their place and time. Pan and zoom the chart; tap the water for its value." />
    </>
  );
}

export function CoastMargin() {
  const { port, key, fromUrl } = useCurrentPort();
  useRememberPort(key, fromUrl);
  return (
    <>
      <ChartScene scene={portScene("coast", port, 30)} />
      <PlaceholderMargin testid="nx-coast" kicker="My Coast" title={port ? `A trip from ${port.name}` : "Plan a trip"} sentence="The gates for your fishery, then the sea window, then a worksheet from your own numbers." />
    </>
  );
}

export function ResearchMargin() {
  const { port, key, fromUrl } = useCurrentPort();
  useRememberPort(key, fromUrl);
  return (
    <>
      <ChartScene scene={portScene("research", port, 45)} />
      <PlaceholderMargin testid="nx-research" kicker="Research" title="The record" sentence={port ? `Station records near ${port.name}: series, quality marks, provenance and export.` : "Station records: series, quality marks, provenance and export."} />
    </>
  );
}

// `/`: home or Today, decided before the first paint (shell/entry.ts) and kept in step on the client
const subscribeNone = () => () => {};

/**
 * What `/` shows. The server cannot know (no cookie), so its HTML holds both, and the pre-paint attribute
 * shows one: a returning reader never sees home first. After hydration only the chosen one stays mounted.
 */
export function RootEntry() {
  const saved = useSavedPortKey();
  const { port } = useCurrentPort();
  const hydrated = useSyncExternalStore(subscribeNone, () => true, () => false);
  const entry: "today" | "home" | null = saved === undefined ? null : saved ? "today" : "home";
  useLayoutEffect(() => {
    if (entry) document.documentElement.setAttribute("data-nx-entry", entry);
  }, [entry]);
  // the scene follows the same decision; before hydration the attribute's choice stands
  const pre = typeof document !== "undefined" ? document.documentElement.getAttribute("data-nx-entry") : null;
  const which = entry ?? (pre === "today" ? "today" : "home");
  return (
    <>
      {hydrated ? <ChartScene scene={which === "today" ? portScene("today", port, 14) : HOME_SCENE} /> : null}
      {entry !== "today" ? <HomeMargin /> : null}
      {entry !== "home" ? (
        <div data-nx-root-today>
          <TodayMargin atRoot />
        </div>
      ) : null}
    </>
  );
}

export { withPlace };
