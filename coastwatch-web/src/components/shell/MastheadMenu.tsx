"use client";

import Link from "next/link";
import { useEffect, useId, useRef, useState } from "react";
import { ModeSwitch } from "@/components/shell/ModeNav";

/**
 * Phones: About, the data status and the presentation switch in one masthead menu, so the tab
 * bar keeps the five places people move between while working (signature P10).
 */
export function MastheadMenu({ active }: { active: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        root.current?.querySelector<HTMLElement>("button")?.focus();
      }
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  const item = "flex min-h-11 items-center justify-between rounded-lg px-3 text-[15px] font-medium text-ink hover:bg-surface-2";
  return (
    <div ref={root} className="relative md:hidden">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        aria-label="Menu: About, data status and presentation"
        data-testid="masthead-menu"
        onClick={() => setOpen(!open)}
        className="cw-press grid h-9 w-9 place-items-center rounded-full text-on-navy hover:bg-white/[0.08]"
      >
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <circle cx="3.5" cy="9" r="1.6" fill="currentColor" />
          <circle cx="9" cy="9" r="1.6" fill="currentColor" />
          <circle cx="14.5" cy="9" r="1.6" fill="currentColor" />
        </svg>
      </button>
      {open && (
        <div id={id} className="theme-paper cw-pop-in absolute right-0 top-[calc(100%+8px)] z-50 w-[min(280px,calc(100vw-24px))] rounded-xl border border-hairline bg-surface p-2 text-ink shadow-[0_10px_30px_rgba(6,17,30,0.3)]" data-testid="masthead-menu-panel">
          <div className="px-3 pb-2 pt-1.5">
            <p className="mb-1.5 text-[13px] text-ink-3">Presentation</p>
            <ModeSwitch tone="paper" />
          </div>
          <Link href="/about" className={item} aria-current={active === "about" ? "page" : undefined} data-testid="menu-about" onClick={() => setOpen(false)}>
            About CoastWatch
          </Link>
          <Link href="/sources" className={item} aria-current={active === "sources" ? "page" : undefined} data-testid="menu-sources" onClick={() => setOpen(false)}>
            Data status and sources
          </Link>
        </div>
      )}
    </div>
  );
}
