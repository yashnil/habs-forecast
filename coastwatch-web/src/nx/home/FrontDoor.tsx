"use client";

import { LayoutGroup, motion, MotionConfig } from "motion/react";
import { useLayoutEffect, useState, useSyncExternalStore } from "react";
import { ChartScene } from "@/nx/chart/scene";
import { Loading } from "@/nx/ds/States";
import { savePort, usePorts, useSavedPortKey } from "@/nx/shell/port";
import { HOME_SCENE } from "@/nx/shell/views";
import { PORT_NAME_LAYOUT, TodayView } from "@/nx/today/TodayView";
import { useFisheries } from "@/nx/today/store";
import type { FinderData } from "./finder.server";
import { Home, type Picked } from "./Home";
import s from "./home.module.css";
import t from "@/nx/today/today.module.css";

const subscribeNone = () => () => {};

/**
 * `/`, the one front door (journeys.md, Architecture). The server cannot know the reader's port (it stays
 * on the device), so the HTML holds both Home and Today's opening lines and the pre-paint entry script
 * (src/nx/shell/entry.ts) shows one: a returning reader never sees Home. After hydration only the chosen
 * one stays. Choosing a port here keeps the page: the name travels from the finder into Today's title, the
 * fishery is asked, then the chart pans and Today assembles in place.
 */
export function FrontDoor({ finder }: { finder: FinderData }) {
  const saved = useSavedPortKey();
  const ports = usePorts();
  const fisheries = useFisheries();
  const hydrated = useSyncExternalStore(subscribeNone, () => true, () => false);
  const [picked, setPicked] = useState<Picked | null>(null);
  const [arriving, setArriving] = useState(false);
  const entry: "today" | "home" | null = saved === undefined ? null : saved ? "today" : "home";

  useLayoutEffect(() => {
    if (entry) document.documentElement.setAttribute("data-nx-entry", entry);
  }, [entry]);
  // the proxy has been drawn where the name was: now the port is saved, and the title takes the name over
  useLayoutEffect(() => {
    if (!picked) return;
    setArriving(true);
    savePort(picked.key);
    setPicked(null);
  }, [picked]);

  const port = saved ? (ports.find((p) => p.key === saved) ?? null) : null;
  const pre = typeof document !== "undefined" ? document.documentElement.getAttribute("data-nx-entry") : null;
  const which = entry ?? (pre === "today" ? "today" : "home");
  const holdCoast = which === "home" || (arriving && !(fisheries && fisheries.length));

  return (
    <MotionConfig reducedMotion="user">
      <LayoutGroup>
        {hydrated && holdCoast ? <ChartScene scene={HOME_SCENE} /> : null}
        {entry !== "today" ? <Home finder={finder} onPick={setPicked} /> : null}
        {picked?.rect ? (
          <motion.span layoutId={PORT_NAME_LAYOUT} className={s.proxy} style={{ left: picked.rect.left, top: picked.rect.top }}>
            {picked.name}
          </motion.span>
        ) : null}
        {entry !== "home" ? (
          <div data-nx-root-today>
            {entry === "today" && port ? (
              <TodayView port={{ key: port.key, name: port.name, lon: port.lon, lat: port.lat }} initial={null} arriving={arriving} />
            ) : (
              <section className={t.today} data-testid="nx-today">
                <p className={t.scope}>
                  <RootPortName />
                </p>
                <Loading>Reading your port&rsquo;s records</Loading>
              </section>
            )}
          </div>
        ) : null}
      </LayoutGroup>
    </MotionConfig>
  );
}

/**
 * Before hydration the port's name is written by a few bytes of script right after its element (the
 * pre-paint entry put the port key on <html>), so a returning reader sees their port from the first frame.
 */
function RootPortName() {
  const ports = usePorts();
  const names = Object.fromEntries(ports.map((p) => [p.key, p.name]));
  const js = `(function(){try{var k=document.documentElement.getAttribute("data-nx-port"),n=${JSON.stringify(names).replace(/</g, "\\u003c")}[k],s=document.currentScript;if(n&&s&&s.previousElementSibling)s.previousElementSibling.textContent=n}catch(e){}})()`;
  return (
    <>
      {/* a text child for the script to replace: React tolerates a changed text under suppressHydrationWarning, not an added one */}
      <span className={t.portName} suppressHydrationWarning>
        {"\u00a0"}
      </span>
      <script suppressHydrationWarning dangerouslySetInnerHTML={{ __html: js }} />
    </>
  );
}
