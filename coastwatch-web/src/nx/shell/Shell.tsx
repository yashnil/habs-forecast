"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";
import { CHART_MODE_KEY, type ChartMode } from "./entry";
import { openPortFinder } from "./finder";
import { FieldPack } from "@/nx/offline/FieldPack";
import { useCurrentPort, useSavedPortKey, type ShellPort } from "./port";

export const DESTINATIONS = [
  { path: "/today", label: "Today" },
  { path: "/explore", label: "Explore" },
  { path: "/coast", label: "My Coast" },
  { path: "/research", label: "Research" },
] as const;

/** The page a pathname belongs to; `/` is Today for a reader with a saved port, else no tab. */
export function activeDestination(pathname: string, savedPort: string | null | undefined): string | null {
  if (pathname === "/") return savedPort ? "/today" : null;
  return DESTINATIONS.find((d) => pathname === d.path || pathname.startsWith(`${d.path}/`))?.path ?? null;
}

/** A tab's address carries the current port, so the port travels across the four pages. */
export const withPlace = (path: string, key: string | null) => (key ? `${path}?place=${encodeURIComponent(key)}` : path);

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
            <Link href="/account#alerts" className="n12-menu-narrow">
              Alerts
            </Link>
            <Link href="/account">Account</Link>
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
 * The shell (README §6): one 52 px bar with the wordmark, the port, the four destinations, Ask and Alerts, and
 * the chart mode in a menu. On a phone the destinations move to a 56 px bottom bar with 48 px targets. Every
 * link carries the current port.
 */
export function Shell() {
  const pathname = usePathname();
  const saved = useSavedPortKey();
  const { port, key } = useCurrentPort();
  const active = activeDestination(pathname, saved);
  const portName = port?.name ?? null;
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
            </Link>
          ))}
        </nav>
        <div className="n12-bar-end">
          <Link href={withPlace("/ask", key)} className="n12-bar-link">
            Ask
          </Link>
          <Link href={key ? `/account?place=${encodeURIComponent(key)}#alerts` : "/account#alerts"} className="n12-bar-link">
            Alerts
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
