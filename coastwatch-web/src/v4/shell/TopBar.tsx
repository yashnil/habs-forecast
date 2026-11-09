"use client";

import Link from "next/link";
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { Lockup, Mark } from "@/v4/brand/Brand";
import { Icon } from "@/v4/ui/Icon";
import { homeHref, readHome } from "@/v4/coast/home";
import { readTodayHref } from "@/pilot/today/navPlace";
import { AlertsBell } from "@/pilot/alerts/AlertsBell";
import { usePilotFrontDoor } from "@/pilot/FrontDoorContext";
import { exploreHref } from "@/pilot/frontDoor";
import { exitProps, usePresence } from "@/v4/motion";
import { TrustPill, type Trust } from "./Notices";
import { useDaylight } from "./useDaylight";

export type Workspace = "today" | "explore" | "coast" | "research" | "page";

const NAV: { id: Exclude<Workspace, "page">; label: string; href: string; icon: string }[] = [
  // the pilot's front door (src/app/today): the reader's port today
  { id: "today", label: "Today", href: "/today", icon: "today" },
  { id: "explore", label: "Explore", href: "/", icon: "map" },
  { id: "coast", label: "My Coast", href: "/coast", icon: "harbor" },
  { id: "research", label: "Research", href: "/research", icon: "research" },
];

// the pilot's own pages (docs/coastwatch/pilot): the reader's ports and alerts, and Ask Neritic
const PILOT_MORE = [
  { label: "Your ports and alerts", href: "/account", testid: "more-account" },
  { label: "Ask Neritic", href: "/ask", testid: "more-ask" },
];

const MORE = [
  // the shore stations' samples and trends live in Research (the v2 bloom page stays at /bloom)
  { label: "Bloom stations", href: "/research" },
  { label: "Fisheries", href: "/fisheries" },
  { label: "Data status", href: "/sources" },
  { label: "About Neritic", href: "/about" },
];

function MoreMenu() {
  const [open, setOpen] = useState(false);
  const [day, setDay] = useDaylight();
  const pop = usePresence(open);
  const box = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      setOpen(false);
    };
    const c = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener("keydown", k);
    window.addEventListener("mousedown", c);
    return () => {
      window.removeEventListener("keydown", k);
      window.removeEventListener("mousedown", c);
    };
  }, [open]);
  const row = "nx-focus nx-row flex min-h-11 items-center rounded-lg px-3 text-[15px] text-n-ink hover:bg-n-surface-2";
  return (
    <div ref={box} className="relative">
      <button type="button" aria-label="More pages" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="nx-press nx-focus nx-tint grid h-11 w-11 place-items-center rounded-xl text-n-ink-2" data-testid="more-menu">
        <Icon name="more" />
      </button>
      {pop.mounted && (
        <nav aria-label="More pages" className="nx-pop absolute right-0 top-[52px] z-50 w-60 rounded-2xl bg-n-surface p-2 shadow-[var(--nx-shadow)]" {...exitProps(pop.state)}>
          {PILOT_MORE.map((m) => (
            <Link key={m.href} href={m.href} onClick={() => setOpen(false)} className={`${row} font-medium`} data-testid={m.testid}>
              {m.label}
            </Link>
          ))}
          <div className="mx-3 my-1 border-t border-n-line" />
          {MORE.map((m) => (
            <Link key={m.label} href={m.href} className={row}>
              {m.label}
            </Link>
          ))}
          <div className="mx-3 my-1 border-t border-n-line" />
          <button
            type="button"
            role="switch"
            aria-checked={day}
            onClick={() => setDay(!day)}
            className="nx-focus nx-row flex min-h-11 w-full items-center justify-between gap-3 rounded-lg px-3 text-left text-[15px] text-n-ink hover:bg-n-surface-2"
            data-testid="daylight-toggle"
          >
            <span>
              Daylight mode
              <span className="block text-[13px] leading-snug text-n-ink-3">Higher contrast for bright sun</span>
            </span>
            <span aria-hidden className={`relative h-6 w-10 shrink-0 rounded-full ${day ? "bg-n-accent" : "bg-n-surface-3"}`}>
              <span className={`absolute top-[3px] h-[18px] w-[18px] rounded-full shadow transition-[left] duration-150 ${day ? "left-[19px] bg-n-on-accent" : "left-[3px] bg-n-ink-2"}`} />
            </span>
          </button>
          <p className="px-3 pb-1.5 pt-2 text-[13px] leading-snug text-n-ink-3">Independent research preview. Not affiliated with NOAA, CDFW or CDPH.</p>
        </nav>
      )}
    </div>
  );
}

/** The phones' sun: one tap to the high-contrast colours (the same switch as in the more menu). */
function SunButton() {
  const [day, setDay] = useDaylight();
  return (
    <button type="button" aria-pressed={day} aria-label="Daylight mode" title="Higher contrast for bright sun" onClick={() => setDay(!day)} className="nx-press nx-focus nx-tint grid h-11 w-11 shrink-0 place-items-center rounded-xl text-n-ink-2 max-[379px]:hidden md:hidden" data-testid="daylight-sun">
      <Icon name="sun" />
    </button>
  );
}

export function NavTabs({ active, size = "bar" }: { active: Workspace; size?: "bar" | "dock" }) {
  // with a home port kept on this device, My Coast opens on its briefing (read after hydration)
  // the whole trip (place, activity, species), so a returning reader lands on their own briefing
  const [home, setHome] = useState<string | null>(null);
  // Today opens on the port the reader last opened there (src/pilot/today/navPlace.ts), read after hydration
  const [todayHref, setTodayHref] = useState<string | null>(null);
  useEffect(() => {
    const h = readHome();
    setHome(h ? homeHref(h) : null);
    setTodayHref(readTodayHref());
  }, []);
  // pilot front door: `/` opens Today, so Explore is linked by its own address
  const pilot = usePilotFrontDoor();
  const nav = useRef<HTMLElement | null>(null);
  useGlidePill(nav, size === "bar");
  return (
    <nav ref={nav} aria-label="Workspaces" className={size === "bar" ? "flex items-center gap-0.5" : "grid grid-cols-4"} data-testid={`nav-${size}`}>
      {NAV.map((n) => {
        const on = n.id === active;
        return (
          <Link
            key={n.id}
            href={n.id === "coast" && home ? home : n.id === "today" && todayHref ? todayHref : n.id === "explore" ? exploreHref(pilot) : n.href}
            aria-current={on ? "page" : undefined}
            data-testid={`nav-${n.id}`}
            className={size === "bar" ? `nx-focus relative isolate flex h-10 items-center rounded-lg px-3 text-[14.5px] font-medium ${on ? "text-n-ink" : "text-n-ink-2 hover:text-n-ink"}` : "nx-dock-item nx-focus"}
          >
            {/* the current workspace's pill (in the server HTML: no flash); it glides from the last workspace's tab */}
            {size === "bar" && on && <span aria-hidden data-nav-pill className="nx-nav-pill" />}
            {size === "dock" && <Icon name={n.icon} className="h-6 w-6" />}
            {n.label}
          </Link>
        );
      })}
    </nav>
  );
}

/** The phone dock: the four workspaces on a soft surface, fixed to the bottom edge. */
export function Dock({ active, absolute = false }: { active: Workspace; absolute?: boolean }) {
  return (
    <div className={`env-instrument nx-dock lg:hidden ${absolute ? "!absolute" : ""}`} data-testid="dock">
      <NavTabs active={active} size="dock" />
    </div>
  );
}

/**
 * Pages remount the top bar, so the pill's last place is kept in sessionStorage; on arrival the pill
 * slides from there to the current tab (transform only, 260 ms). Reduced motion: it is simply there.
 */
function useGlidePill(nav: React.RefObject<HTMLElement | null>, on: boolean) {
  useLayoutEffect(() => {
    const box = nav.current;
    const pill = box?.querySelector<HTMLElement>("[data-nav-pill]");
    if (!on || !box || !pill) return;
    const b = box.getBoundingClientRect();
    const p = pill.getBoundingClientRect();
    const now = { x: p.left - b.left, w: p.width };
    try {
      const prev = JSON.parse(sessionStorage.getItem("nx-nav-pill-v1") ?? "null") as { x: number; w: number } | null;
      sessionStorage.setItem("nx-nav-pill-v1", JSON.stringify(now));
      if (!prev || (Math.abs(prev.x - now.x) < 1 && Math.abs(prev.w - now.w) < 1)) return;
      if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      pill.style.transformOrigin = "left center";
      pill.animate([{ transform: `translateX(${prev.x - now.x}px) scaleX(${prev.w / now.w})` }, { transform: "none" }], { duration: 260, easing: "cubic-bezier(0.23, 1, 0.32, 1)" });
    } catch {
      // storage unavailable: no glide
    }
  }, [nav, on]);
}

/** Has the page scrolled past `y`? Only listens when asked. */
function useScrolled(on: boolean, y = 24) {
  const [past, setPast] = useState(false);
  useEffect(() => {
    if (!on) {
      setPast(false);
      return;
    }
    const f = () => setPast(window.scrollY > y);
    f();
    window.addEventListener("scroll", f, { passive: true });
    return () => window.removeEventListener("scroll", f);
  }, [on, y]);
  return past;
}

/**
 * One frame for every page (design.md §7.1): brand, the four workspaces (wide screens; the dock on
 * phones), a slot (search in Explore), the trust pill, the alerts bell and the more menu.
 *
 * `over`: the bar sits over a stage: transparent with a soft top scrim, and it takes the solid ground once
 * the page has scrolled 24 px. It adds no height of its own (the stage starts at the top of the page).
 * `floating`: the bar sits over a canvas that does not scroll (Explore's map): absolute, always the scrim.
 * Neither: sticky, solid ground (the default).
 * `trust`: what the pill says; `loadTrust()` (src/lib/data.ts) provides it on the server. The pill and its
 * words are always there, even before the data arrives or when it fails.
 */
// z-[45]: the header is a stacking context, so its menus and the notices panel can rise no higher
// than it; it sits above the phone sheet (z-30) and dock (z-40), below modal dialogs (z-50).
export function TopBar({ active, now, slot, trust = null, over = false, floating = false }: { active: Workspace; now: Date | null; slot?: ReactNode; trust?: Trust | null; over?: boolean; floating?: boolean }) {
  const scrolled = useScrolled(over && !floating);
  const overStage = over || floating;
  return (
    <header
      className={`nx-topbar z-[45] ${floating ? "pointer-events-none absolute inset-x-0 top-0" : "sticky top-0"}`}
      data-over={overStage ? "" : undefined}
      data-solid={overStage && scrolled ? "" : undefined}
      style={over && !floating ? { marginBottom: "calc(-1 * var(--nx-bar-h))" } : undefined}
    >
      <Link href="/" aria-label="Neritic home" className="nx-press nx-focus pointer-events-auto flex h-11 shrink-0 items-center rounded-lg text-n-ink lg:pr-1" data-testid="brand-home">
        <span className="max-md:hidden">
          <Lockup size={26} />
        </span>
        <span className="md:hidden">
          <Mark size={30} />
        </span>
      </Link>
      <div className="pointer-events-auto hidden lg:block">
        <NavTabs active={active} />
      </div>
      {/* on a phone the slot is what it needs (Explore: the search icon); the pill takes the rest */}
      <div className={`pointer-events-auto min-w-0 flex-1 ${slot ? "max-md:min-w-11 max-md:flex-none" : ""}`}>{slot}</div>
      <div className="pointer-events-auto flex min-w-0 items-center gap-0.5 max-md:ml-auto" data-testid="topbar-cluster">
        <TrustPill trust={trust} now={now} />
        {!slot && <SunButton />}
        {/* pilot: alerts for the reader's saved ports (the device inbox; the account's too when signed in) */}
        <AlertsBell />
        <MoreMenu />
      </div>
    </header>
  );
}
