"use client";

import Link from "next/link";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { SourceStatus } from "@/generated/schema";
import type { OfficialDataset, OfficialRecord } from "@/generated/official";
import { OFFICIAL_STATUS } from "@/content/copy";
import { activeRecords, officialVerification, type Verification } from "@/lib/official";
import { useNow } from "@/lib/useNow";
import { Hotlines, NoticeCard, VerificationBadge, VerificationDetail } from "@/components/official/Official";
import { SourceLink } from "@/components/ui/Primitives";
import { Icon } from "@/components/ui/Icon";

/**
 * Official notices are global: every page shows their count and verification state in the
 * masthead (and in the mobile tab bar), and one tap opens the full list. Design reset §4.1.
 */
type Ctx = {
  ds: OfficialDataset | null;
  records: OfficialRecord[];
  verification: Verification | null;
  now: Date | null;
  open: boolean;
  openDrawer: (opener?: HTMLElement | null) => void;
  closeDrawer: () => void;
};

const OfficialCtx = createContext<Ctx | null>(null);

/** Open the global official-notices drawer from anywhere inside the shell. */
export function useOfficialDrawer() {
  const { openDrawer, records, verification, ds } = useOfficial();
  return { openDrawer, count: records.length, verification, available: !!ds };
}

function useOfficial(): Ctx {
  const c = useContext(OfficialCtx);
  if (!c) throw new Error("useOfficial outside OfficialProvider");
  return c;
}

export function OfficialProvider({ official, status, children }: { official: OfficialDataset | null; status: SourceStatus | null; children: ReactNode }) {
  const now = useNow();
  const [open, setOpen] = useState(false);
  const opener = useRef<HTMLElement | null>(null);
  const records = useMemo(() => activeRecords(official), [official]);
  const verification = useMemo(() => (now ? officialVerification(official, status, now) : null), [official, status, now]);
  const openDrawer = useCallback((el?: HTMLElement | null) => {
    opener.current = el ?? (document.activeElement as HTMLElement | null);
    setOpen(true);
  }, []);
  const closeDrawer = useCallback(() => {
    setOpen(false);
    // give focus back to whatever opened the drawer
    requestAnimationFrame(() => opener.current?.focus());
  }, []);
  const value = useMemo(() => ({ ds: official, records, verification, now, open, openDrawer, closeDrawer }), [official, records, verification, now, open, openDrawer, closeDrawer]);
  return <OfficialCtx.Provider value={value}>{children}</OfficialCtx.Provider>;
}

/** Plain-text verification word for compact places (pill, tab badge). */
export function verificationWord(v: Verification | null): string {
  return v ? OFFICIAL_STATUS.verification[v.state] : "Checking…";
}

/** Masthead control: count of active notices and the registry's verification state. */
export function OfficialPill() {
  const { ds, records, verification, open, openDrawer } = useOfficial();
  return (
    <button
      type="button"
      data-testid="official-pill"
      data-state={verification?.state ?? "checking"}
      aria-haspopup="dialog"
      aria-expanded={open}
      onClick={(e) => openDrawer(e.currentTarget)}
      className="flex h-8 items-center gap-2 rounded-full border border-official-on-navy/45 bg-official-on-navy/10 pl-2.5 pr-3 text-[13px] font-medium text-official-on-navy hover:bg-official-on-navy/15 md:h-[34px] md:text-[14px]"
    >
      <Icon name="shield" className="h-4 w-4" />
      {ds ? (
        <>
          <b className="font-semibold text-white tabular">{records.length}</b>
          <span className="hidden sm:inline">official notice{records.length === 1 ? "" : "s"}</span>
        </>
      ) : (
        <span>Official notices</span>
      )}
      <span className="h-3.5 w-px bg-official-on-navy/40" aria-hidden />
      <span>{ds ? verificationWord(verification) : "Unavailable"}</span>
    </button>
  );
}

const AGENCY_NAME: Record<string, string> = {
  CDFW: "California Department of Fish and Wildlife",
  CDPH: "California Department of Public Health",
  OEHHA: "Office of Environmental Health Hazard Assessment",
};

/** Full list of official notices, opened from the pill or the Notices tab. Paper surface on every page. */
export function OfficialDrawer() {
  const { ds, records, verification, now, open, closeDrawer } = useOfficial();
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const el = panel.current;
    el?.querySelector<HTMLElement>("[data-autofocus]")?.focus();
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeDrawer();
      if (e.key !== "Tab" || !el) return;
      // keep focus inside the dialog
      const f = [...el.querySelectorAll<HTMLElement>("a[href],button:not([disabled]),summary,[tabindex]:not([tabindex='-1'])")].filter((n) => n.offsetParent !== null);
      if (!f.length) return;
      const first = f[0], last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, closeDrawer]);

  // Not rendered while closed, so notice ids and test hooks exist once on the page.
  if (!open) return null;
  const agencies = [...new Set(records.map((r) => r.agency))];
  return (
    <>
      <div className="fixed inset-0 z-50 bg-navy-950/45" onClick={closeDrawer} aria-hidden data-testid="official-drawer-scrim" />
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby="official-drawer-h"
        data-testid="official-drawer"
        className="fixed inset-y-0 right-0 z-50 flex w-full max-w-[520px] flex-col bg-surface text-ink shadow-[0_8px_32px_rgba(6,17,30,0.3)]"
      >
        <header className="relative border-b border-official-line bg-official-bg px-6 pb-4 pt-6">
          <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-official">{OFFICIAL_STATUS.heading}</p>
          <h2 id="official-drawer-h" className="mt-1.5 font-display text-[24px] font-medium leading-tight">
            {ds ? `${records.length} active notice${records.length === 1 ? "" : "s"} in California` : "Official notices unavailable"}
          </h2>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {ds && <VerificationBadge v={verification} />}
            <span className="text-[13px] text-ink-2">The agency pages are the authority.</span>
          </div>
          <button
            type="button"
            data-autofocus
            onClick={closeDrawer}
            aria-label="Close official notices"
            className="absolute right-4 top-4 grid h-9 w-9 place-items-center rounded-lg text-ink-2 hover:bg-surface"
          >
            <Icon name="close" className="h-5 w-5" />
          </button>
        </header>
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-6 pb-8 pt-4">
          {verification && verification.state !== "verified" && ds && <VerificationDetail v={verification} />}
          {!ds && (
            <p className="text-[14px] text-ink-2">
              {OFFICIAL_STATUS.notTracked} {OFFICIAL_STATUS.instruction}
            </p>
          )}
          {agencies.map((a) => (
            <section key={a} aria-labelledby={`drawer-${a}`} className="space-y-1.5">
              <h3 id={`drawer-${a}`} className="text-[12px] font-semibold uppercase tracking-[0.08em] text-ink-3">
                {AGENCY_NAME[a] ?? a} · {records.filter((r) => r.agency === a).length}
              </h3>
              {records
                .filter((r) => r.agency === a)
                .map((r) => (
                  <NoticeCard key={r.id} r={r} now={now} compact />
                ))}
            </section>
          ))}
          {ds && ds.registry.statements.length > 0 && (
            <details className="rounded-md border border-hairline px-3 py-2 text-[13px] text-ink-2">
              <summary className="cursor-pointer font-medium text-ink">{OFFICIAL_STATUS.statementsHeading}</summary>
              <p className="mt-1 text-[12px] text-ink-3">{OFFICIAL_STATUS.statementsNote}</p>
              <ul className="mt-1.5 space-y-1.5">
                {ds.registry.statements.map((s) => (
                  <li key={s.id}>
                    <p className="text-[12px] text-ink-3">
                      {s.agency} · {s.topic}
                    </p>
                    <p>
                      “{s.statement}” <SourceLink href={s.source.url}>source ↗</SourceLink>
                    </p>
                  </li>
                ))}
              </ul>
            </details>
          )}
          <p className="text-[13px] leading-snug text-ink-2">{OFFICIAL_STATUS.missingNotOpen}</p>
          <ul className="space-y-1 text-[13px]">
            {OFFICIAL_STATUS.links.map((l) => (
              <li key={l.href}>
                <SourceLink href={l.href}>{l.label} ↗</SourceLink>
              </li>
            ))}
          </ul>
          <div className="rounded-md border border-hairline bg-surface-2 px-3 py-2.5">
            <Hotlines />
          </div>
        </div>
      </div>
    </>
  );
}

const TABS = [
  { key: "map", href: "/", label: "Map", icon: "map" },
  { key: "bloom", href: "/bloom", label: "Blooms", icon: "bloom" },
  { key: "fisheries", href: "/fisheries", label: "Fisheries", icon: "fish" },
] as const;

/** Mobile bottom navigation. Notices is a tab, so official information is one tap from anywhere. */
export function TabBar({ active }: { active: string }) {
  const { ds, records, open, openDrawer } = useOfficial();
  return (
    <nav
      aria-label="Primary"
      data-testid="tabbar"
      className="fixed inset-x-0 bottom-0 z-40 grid h-[var(--cw-tabbar-h)] grid-cols-4 border-t border-hairline bg-surface pb-[env(safe-area-inset-bottom)] text-ink-3 md:hidden"
      style={{ colorScheme: "light" }}
    >
      {TABS.map((t) => (
        <Link
          key={t.key}
          href={t.href}
          data-testid={`tab-${t.key}`}
          aria-current={active === t.key ? "page" : undefined}
          className={`relative flex flex-col items-center justify-center gap-1 text-[12px] font-medium ${active === t.key ? "text-accent before:absolute before:inset-x-[30%] before:top-0 before:h-0.5 before:rounded-b before:bg-accent" : ""}`}
        >
          <Icon name={t.icon} className="h-[22px] w-[22px]" />
          {t.label}
        </Link>
      ))}
      <button
        type="button"
        data-testid="tab-notices"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={(e) => openDrawer(e.currentTarget)}
        className="relative flex flex-col items-center justify-center gap-1 text-[12px] font-medium text-official"
      >
        <Icon name="shield" className="h-[22px] w-[22px]" />
        {ds && (
          <span className="absolute left-[calc(50%+6px)] top-2 min-w-[18px] rounded-full bg-official px-1.5 text-center text-[11px] font-semibold leading-[18px] text-white tabular">
            {records.length}
            <span className="sr-only"> active</span>
          </span>
        )}
        Notices
      </button>
    </nav>
  );
}
