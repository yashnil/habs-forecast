"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";
import { CHART_MODE_KEY, type ChartMode } from "./entry";
import { openPortFinder } from "./finder";
import { FieldPack } from "@/nx/offline/FieldPack";
import { useCurrentPort, useSavedPortKey, type ShellPort } from "./port";
import { DESTINATIONS, activeDestination, withPlace } from "./destinations";
import { alertsHref } from "./unread";
import { useUnread } from "@/nx/alerts/useInbox";
import { motion } from "motion/react";
import { shared } from "@/nx/ds/motion";

export { DESTINATIONS, activeDestination, withPlace } from "./destinations";

function Chevron() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden="true" className="n12-chev">
      <path d="M6 9l6 6 6-6" fill="none" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

function ChartModeMenu({ port }: { port: ShellPort | null }) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<ChartMode>("day");
  const ref = useRef<HTMLDivElement>(null);
  const id = useId();
  useEffect(() => {
    setMode((document.documentElement.getAttribute("data-chart-mode") as ChartMode) || "day");
  }, []);
  useEffect(() => {
    if (!open) return;
    const off = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", off);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("pointerdown", off);
      document.removeEventListener("keydown", key);
    };
  }, [open]);
  const choose = (m: ChartMode) => {
    setMode(m);
    document.documentElement.setAttribute("data-chart-mode", m);
    try {
      localStorage.setItem(CHART_MODE_KEY, m);
    } catch {
      // this page view only
    }
  };
  const modes: Array<[ChartMode, string, string]> = [
    ["day", "Day", "Sunlight"],
    ["dusk", "Dusk", "Evening wheelhouse"],
    ["night", "Night", "Dark-adapted, very dim"],
  ];
  return (
    <div className="n12-menu" ref={ref}>
      <button type="button" className="n12-bar-link" aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)} data-testid="nx-menu-button">
        Menu
      </button>
      {open ? (
        <div className="n12-menu-panel" id={id}>
          <fieldset className="n12-mode">
            <legend>Chart</legend>
            {modes.map(([m, label, hint]) => (
              <label key={m} className="n12-mode-option">
                <input type="radio" name={`${id}-mode`} value={m} checked={mode === m} onChange={() => choose(m)} data-testid={`nx-mode-${m}`} />
                <span>
                  {label} <small>{hint}</small>
                </span>
              </label>
            ))}
          </fieldset>
          <FieldPack port={port ? { key: port.key, name: port.name, lat: port.lat, lon: port.lon } : null} />
          <nav className="n12-menu-links" aria-label="More">
            <Link href="/ask" className="n12-menu-narrow">
              Ask
            </Link>
            <Link href={alertsHref(port?.key ?? null)} className="n12-menu-narrow">
              Alerts
            </Link>
            <Link href="/account">Your data</Link>
            <Link href="/sources">Sources</Link>
            <Link href="/about">About</Link>
            <Link href="/privacy">Privacy</Link>
          </nav>
          <p className="n12-menu-note">
            Chart: depth from NOAA ETOPO 2022 and the Coastal Relief Model; coast © OpenStreetMap contributors, OpenMapTiles. Not for navigation.
          </p>
        </div>
      ) : null}
    </div>
  );
}

/**
 * The shell (README §6): one 52 px bar with the wordmark, the port, the four destinations, Ask and Alerts (with
 * the unread count when there is one, src/nx/alerts/useInbox), and the chart mode in a menu. On a phone the destinations move to a 56 px bottom bar with 48 px targets. Every
 * link carries the current port.
 */
export function Shell() {
  const pathname = usePathname();
  const saved = useSavedPortKey();
  const { port, key } = useCurrentPort();
  const active = activeDestination(pathname, saved);
  const portName = port?.name ?? null;
  // undefined until the device's inbox has been read (after hydration): nothing is shown until then
  const unread = useUnread() ?? 0;
  // the travelling underline is drawn once the app has hydrated (the saved port, which decides the active
  // destination at `/`, is known only then); until then the stylesheet underlines the current link
  const [live, setLive] = useState(false);
  useEffect(() => setLive(true), []);
  return (
    <>
      <a className="n12-skip" href="#nx-margin">
        Skip to content
      </a>
      <header className="n12-bar" data-testid="nx-bar">
        <Link href="/" className="n12-wordmark" aria-label="Neritic, home">
          Neritic
        </Link>
        <button type="button" className="n12-port-control" onClick={openPortFinder} data-testid="nx-port-control" aria-label={portName ? `Port: ${portName}. Change port` : "Find your port"}>
          <span className="n12-port-control-name" suppressHydrationWarning>
            {portName ?? "Find your port"}
          </span>
          <Chevron />
        </button>
        <nav className="n12-dest" aria-label="Destinations">
          {DESTINATIONS.map((d) => (
            <Link key={d.path} href={withPlace(d.path, key)} aria-current={active === d.path ? "page" : undefined} className="n12-dest-link">
              {d.label}
              {/* one underline that travels to the new destination (`shared`) */}
              {live && active === d.path ? <motion.span layoutId="nx-dest" className="n12-dest-on" transition={shared} aria-hidden="true" /> : null}
            </Link>
          ))}
        </nav>
        <div className="n12-bar-end">
          <Link href={withPlace("/ask", key)} className="n12-bar-link">
            Ask
          </Link>
          <Link href={alertsHref(key)} className="n12-bar-link" data-testid="nx-alerts-link" aria-label={unread ? `Alerts, ${unread} unread` : undefined}>
            Alerts
            {unread ? (
              <span className="n12-unread" data-testid="nx-unread" aria-hidden="true">
                {unread > 99 ? "99+" : unread}
              </span>
            ) : null}
          </Link>
          <ChartModeMenu port={port} />
        </div>
      </header>
      <nav className="n12-tabs" aria-label="Destinations" data-testid="nx-tabs">
        {DESTINATIONS.map((d) => (
          <Link key={d.path} href={withPlace(d.path, key)} aria-current={active === d.path ? "page" : undefined} className="n12-tab">
            {d.label}
          </Link>
        ))}
      </nav>
    </>
  );
}
