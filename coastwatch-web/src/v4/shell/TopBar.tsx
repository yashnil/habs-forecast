"use client";

import Link from "next/link";
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { Lockup, Mark } from "@/v4/brand/Brand";
import { Icon } from "@/v4/ui/Icon";
import { StatusCenter } from "@/v4/readiness/StatusCenter";
import { readHomePlace } from "@/v4/state/homePort";
import { exitProps, usePresence } from "@/v4/motion";

export type Workspace = "explore" | "coast" | "research" | "page";

const NAV: { id: Exclude<Workspace, "page">; label: string; href: string; icon: string }[] = [
  { id: "explore", label: "Explore", href: "/", icon: "map" },
  { id: "coast", label: "My Coast", href: "/coast", icon: "harbor" },
  { id: "research", label: "Research", href: "/research", icon: "research" },
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
  return (
    <div ref={box} className="relative">
      <button type="button" aria-label="More pages" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="nx-press nx-focus grid h-11 w-11 place-items-center rounded-full border border-n-line-strong bg-n-surface text-n-ink-2" data-testid="more-menu">
        <Icon name="more" />
      </button>
      {pop.mounted && (
        <nav aria-label="More pages" className="nx-pop absolute right-0 top-[52px] z-50 w-56 rounded-[18px] border border-n-line bg-n-surface p-1.5 shadow-[var(--nx-shadow)]" {...exitProps(pop.state)}>
          {MORE.map((m) => (
            <Link key={m.href} href={m.href} className="nx-focus nx-row flex min-h-11 items-center rounded-lg px-3 text-[15px] text-n-ink hover:bg-n-surface-2">
              {m.label}
            </Link>
          ))}
          <p className="px-3 pb-1.5 pt-2 text-[12px] leading-snug text-n-ink-3">Independent research preview. Not affiliated with NOAA, CDFW or CDPH.</p>
        </nav>
      )}
    </div>
  );
}

export function NavTabs({ active, size = "bar" }: { active: Workspace; size?: "bar" | "dock" }) {
  // with a home port kept on this device, My Coast opens on its briefing (read after hydration)
  const [home, setHome] = useState<string | null>(null);
  useEffect(() => setHome(readHomePlace()), []);
  const nav = useRef<HTMLElement | null>(null);
  useGlidePill(nav, size === "bar");
  return (
    <nav ref={nav} aria-label="Workspaces" className={size === "bar" ? "flex items-center gap-1 rounded-full border border-n-line bg-n-surface p-1" : "grid grid-cols-3"} data-testid={`nav-${size}`}>
      {NAV.map((n) => {
        const on = n.id === active;
        return (
          <Link
            key={n.id}
            href={n.id === "coast" && home ? `/coast?place=${encodeURIComponent(home)}` : n.href}
            aria-current={on ? "page" : undefined}
            data-testid={`nav-${n.id}`}
            className={
              size === "bar"
                ? `nx-press nx-focus relative isolate flex h-9 items-center rounded-full px-3.5 text-[14px] font-semibold ${on ? "text-n-bg" : "text-n-ink-2 hover:text-n-ink"}`
                : `nx-focus flex min-h-[60px] flex-col items-center justify-center gap-0.5 text-[12.5px] font-semibold ${on ? "text-n-accent" : "text-n-ink-2"}`
            }
          >
            {/* the active pill is in the server HTML (no flash); it glides from the last workspace's tab */}
            {size === "bar" && on && <span aria-hidden data-nav-pill className="absolute inset-0 -z-10 rounded-full bg-n-ink" />}
            {size === "dock" && <Icon name={n.icon} className="h-6 w-6" />}
            {n.label}
          </Link>
        );
      })}
    </nav>
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

/**
 * The top bar: floating over the map in Explore, a band over reading pages. Brand (C2),
 * workspaces, a slot (search in Explore), the status centre, official notices, more.
 */
// z-[45]: the header is a stacking context, so its menus and the notices panel can rise no higher
// than it; it sits above the phone sheet (z-30) and dock (z-40), below modal dialogs (z-50).
export function TopBar({ active, now, slot, notices, floating = false }: { active: Workspace; now: Date | null; slot?: ReactNode; notices?: ReactNode; floating?: boolean }) {
  return (
    <header
      className={`pointer-events-none z-[45] flex items-start gap-2 px-3 pt-3 md:gap-3 md:px-4 ${floating ? "absolute inset-x-0 top-0" : "relative pb-3"}`}
      style={{ paddingTop: "max(12px, env(safe-area-inset-top, 0px))" }}
    >
      <Link href="/" aria-label="Neritic home" className="nx-focus pointer-events-auto flex h-11 shrink-0 items-center rounded-full border border-n-line bg-n-surface pl-1.5 pr-3.5 text-n-ink max-md:pr-1.5" data-testid="brand-home">
        <span className="max-md:hidden">
          <Lockup size={30} />
        </span>
        <span className="md:hidden">
          <Mark size={32} />
        </span>
      </Link>
      <div className="pointer-events-auto hidden lg:block">
        <NavTabs active={active} />
      </div>
      <div className="pointer-events-auto min-w-0 flex-1">{slot}</div>
      <div className="pointer-events-auto flex shrink-0 items-start gap-2">
        {notices}
        <span className="max-sm:hidden">
          <StatusCenter now={now} />
        </span>
        <span className="sm:hidden">
          <StatusCenter now={now} compact />
        </span>
        <MoreMenu />
      </div>
    </header>
  );
}
