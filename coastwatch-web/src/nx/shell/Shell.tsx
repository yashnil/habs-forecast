"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";
import { motion } from "motion/react";
import { CHART_MODE_KEY, type ChartMode } from "./entry";
import { openPortFinder } from "./finder";
import { FieldPack } from "@/nx/offline/FieldPack";
import { useCurrentPort, useSavedPortKey, type ShellPort } from "./port";
import { DESTINATIONS, activeDestination, withPlace } from "./destinations";
import { alertsHref } from "./unread";
import { useUnread } from "@/nx/alerts/useInbox";
import { Icon } from "@/nx/ds/Icon";
import { navPill } from "@/nx/ds/motion";
import { LayerChips } from "@/nx/ds/synth/Chips";
import { usePortLabel } from "./portLabel";

export { DESTINATIONS, activeDestination, withPlace } from "./destinations";

/** The brand's mark: three swells, the nearest strongest (from the approved synthesis). */
function BrandMark() {
  return (
    <svg className="nx-brand-mark" viewBox="0 0 28 28" aria-hidden="true">
      <path d="M3 22 C 9 22, 13 18, 13 12 S 17 3, 25 3" />
      <path d="M9 25 C 15 25, 19 21, 19 15 S 21 10, 25 10" opacity=".55" />
      <path d="M17 25 C 21 25, 24 22, 25 18" opacity=".28" />
    </svg>
  );
}

function Chevron() {
  return (
    <svg className="nx-port-chev" viewBox="0 0 10 6" aria-hidden="true">
      <path d="M1 1l4 4 4-4" />
    </svg>
  );
}

function ChartModeMenu({ port }: { port: ShellPort | null }) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<ChartMode>("day");
  const ref = useRef<HTMLDivElement>(null);
  const id = useId();
  const { key } = useCurrentPort();
  useEffect(() => {
    setMode((document.documentElement.getAttribute("data-chart-mode") as ChartMode) || "day");
  }, []);
  // the mode may have been changed elsewhere (Your data) since this menu last opened: read it each time it opens
  useEffect(() => {
    if (open) setMode((document.documentElement.getAttribute("data-chart-mode") as ChartMode) || "day");
  }, [open]);
  useEffect(() => {
    if (!open) return;
    const off = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const keyDown = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", off);
    document.addEventListener("keydown", keyDown);
    return () => {
      document.removeEventListener("pointerdown", off);
      document.removeEventListener("keydown", keyDown);
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
      <button type="button" className="nx-menu-btn" aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)} data-testid="nx-menu-button" aria-label="Menu" title="Menu">
        <Icon name="menu" size={20} />
        <span className="nx-menu-word">Menu</span>
      </button>
      {open ? (
        <div className="n12-menu-panel" id={id}>
          <nav className="n12-menu-links" aria-label="More">
            <Link href={withPlace("/explore", key)}>Explore the water</Link>
            <Link href={withPlace("/ask", key)}>Ask</Link>
            <Link href="/account">Your data</Link>
            <Link href="/sources">Sources</Link>
            <Link href="/about">About</Link>
            <Link href="/privacy">Privacy</Link>
          </nav>
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
          <p className="n12-menu-note">
            Chart: depth from NOAA ETOPO 2022 and the Coastal Relief Model; coast © OpenStreetMap contributors, OpenMapTiles. Not for navigation.
          </p>
        </div>
      ) : null}
    </div>
  );
}

/** One section link; the current one carries the dark pill, which slides between sections on a spring. */
function Section({ path, label, href, current, unread, live, pillId, testid }: { path: string; label: string; href: string; current: boolean; unread: number; live: boolean; pillId: string; testid?: string }) {
  const alerts = path === "/alerts";
  return (
    <Link href={href} aria-current={current ? "page" : undefined} className="n12-dest-link" data-testid={testid} aria-label={alerts && unread ? `Alerts, ${unread} unread` : undefined}>
      {live && current ? <motion.span layoutId={pillId} className="n12-dest-on" transition={navPill} aria-hidden="true" /> : null}
      <span className="nx-dest-word">{label}</span>
      {alerts && unread ? (
        <span className="nx-count" data-testid={testid ? "nx-unread" : undefined} aria-hidden="true">
          {unread > 99 ? "99+" : unread}
        </span>
      ) : null}
    </Link>
  );
}

/**
 * The shell (v14 README §2): a floating top bar over the chart. On the left the brand pill (the mark, "Neritic", and
 * the port with its finder); on the right the nav pill (Today / My Coast / Research / Alerts with the unread count, a
 * dark pill that slides to the current section) and the menu. Under the brand, the layer chips. On a phone the
 * sections move to a bottom tab bar and the chips become one "Layers" button. Every link carries the current port.
 */
export function Shell() {
  const pathname = usePathname();
  const saved = useSavedPortKey();
  const { port, key } = useCurrentPort();
  const active = activeDestination(pathname, saved);
  const portName = port?.name ?? null;
  // a page about a station names its area ("Near Santa Cruz"); otherwise the port
  const context = usePortLabel();
  const pill = context ?? portName;
  // undefined until the device's inbox has been read (after hydration): nothing is shown until then
  const unread = useUnread() ?? 0;
  // the sliding pill is drawn once hydrated (the saved port, which decides the section at `/`, is known only then);
  // until then the stylesheet fills the current link
  const [live, setLive] = useState(false);
  useEffect(() => setLive(true), []);
  const href = (path: string) => (path === "/alerts" ? alertsHref(key) : withPlace(path, key));
  return (
    <>
      <a className="n12-skip" href="#nx-margin">
        Skip to content
      </a>
      <header className="n12-bar" data-testid="nx-bar">
        <div className="nx-brand">
          <Link href="/" className="nx-brand-home n12-wordmark" aria-label="Neritic, home">
            <BrandMark />
            <span className="nx-brand-name">Neritic</span>
          </Link>
          <button type="button" className="nx-brand-port" onClick={openPortFinder} data-testid="nx-port-control" aria-haspopup="dialog" aria-label={context ? `${context}. Choose a port` : portName ? `Port: ${portName}. Change port` : "Find your port"}>
            <span className="n12-port-control-name" suppressHydrationWarning>
              {pill ?? "Find your port"}
            </span>
            <Chevron />
          </button>
        </div>
        <div className="nx-bar-end">
          <nav className="nx-nav" aria-label="Destinations">
            {DESTINATIONS.map((d) => (
              <Section key={d.path} path={d.path} label={d.label} href={href(d.path)} current={active === d.path} unread={unread} live={live} pillId="nx-dest" testid={d.path === "/alerts" ? "nx-alerts-link" : undefined} />
            ))}
          </nav>
          <ChartModeMenu port={port} />
        </div>
      </header>
      <LayerChips className="nx-layers" />
      <nav className="n12-tabs" aria-label="Destinations" data-testid="nx-tabs">
        {DESTINATIONS.map((d) => (
          <Section key={d.path} path={d.path} label={d.label} href={href(d.path)} current={active === d.path} unread={unread} live={live} pillId="nx-tab" />
        ))}
      </nav>
    </>
  );
}
