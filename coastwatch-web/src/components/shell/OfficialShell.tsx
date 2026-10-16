"use client";

import Link from "next/link";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { SourceStatus } from "@/generated/schema";
import type { OfficialDataset, OfficialRecord } from "@/generated/official";
import { OFFICIAL_STATUS } from "@/content/copy";
import { currentRecords, disputedRecords, officialVerification, type Verification } from "@/lib/official";
import { useNow } from "@/lib/useNow";
import { Hotlines, NoticeCard, VerificationBadge, VerificationDetail } from "@/components/official/Official";
import { SourceLink } from "@/components/ui/Primitives";
import { Icon } from "@/components/ui/Icon";
import { Glide } from "@/components/motion/Glide";

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
  // a record that an agency release appears to lift is not counted as a current notice
  const records = useMemo(() => currentRecords(official), [official]);
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

export const CDFW_URL = "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories";
export const CDPH_URL = "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx";

/**
 * Site-wide disclosure while the notice registry is not human-verified: says why (agency pages
 * changed since the last review, with the unreviewed items, or never checked by a person) and
 * that the list may be incomplete. Driven by the verification state, so it appears for any
 * future change too and disappears only after a recorded human review. Rendered by the
 * status line (components/Banners.tsx).
 */
export function useRegistryDisclosure(): { show: boolean; pending: boolean; changed: boolean; reason: string; openDrawer: (el?: HTMLElement | null) => void } {
  const { ds, verification, openDrawer } = useOfficial();
  const changed = (ds?.watch ?? []).filter((w) => w.ok && (w.new_items?.length || w.matches_review === false));
  const items = changed.flatMap((w) => w.new_items ?? []);
  const reason = !ds
    ? "Official notices could not be loaded."
    : changed.length
      ? `Agency pages have changed since this list was last reviewed${items.length ? ` (not yet reviewed: ${items.join(", ")})` : ""}. The list may be incomplete or out of date.`
      : "This list was transcribed from the agency pages and has not been checked by a person. It may be incomplete.";
  // pending: the state is computed in the browser (it depends on today's date); until then the
  // status line keeps its place so the page does not jump when it appears
  return { show: !!verification && verification.state !== "verified", pending: !verification, changed: changed.length > 0, reason, openDrawer };
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
      className="flex h-8 shrink-0 items-center gap-2 whitespace-nowrap rounded-full border border-white/20 bg-white/[0.06] pl-3 pr-3 min-[360px]:pl-2.5 text-[13px] font-medium text-on-navy transition-colors hover:border-white/35 hover:bg-white/10 md:h-[34px] md:text-[14px]"
    >
      {/* below 360 px the masthead keeps the count and the verification word, not the icon */}
      <Icon name="shield" className="h-4 w-4 text-official-on-navy max-[359px]:hidden" />
      {ds ? (
        <>
          <b className="font-semibold text-white tabular">{records.length}</b>
          {/* the words show only where the masthead has room; screen readers always hear them */}
          <span className="hidden xl:inline">
            <span className="sr-only">agency </span>notice{records.length === 1 ? "" : "s"}
          </span>
          <span className="sr-only xl:hidden">agency notice{records.length === 1 ? "" : "s"}</span>
        </>
      ) : (
        <span>Official notices</span>
      )}
      <span className="h-3.5 w-px bg-white/25" aria-hidden />
      {/* the words say whose check is missing: the agencies published the notices; Neritic's copy of them is unreviewed */}
      <span className={verification && verification.state !== "verified" ? "text-on-navy-2" : ""}>{ds ? (verification && verification.state !== "verified" && verification.state !== "unavailable" ? <>Not verified<span className="max-2xl:hidden"> copy</span></> : verificationWord(verification)) : "Unavailable"}</span>
    </button>
  );
}

/** The drawer groups notices by what people harvest, from each record's own species list. */
const GROUPS = [
  { id: "shellfish", label: "Shellfish: mussels, clams, oysters, scallops" },
  { id: "crab", label: "Crab" },
  { id: "fish", label: "Fish" },
  { id: "other", label: "Other" },
] as const;
function groupOf(r: OfficialRecord): (typeof GROUPS)[number]["id"] {
  const s = r.species.join(" ").toLowerCase();
  if (/mussel|clam|oyster|scallop|bivalve/.test(s)) return "shellfish";
  if (/crab|lobster/.test(s)) return "crab";
  if (/anchov|sardine|salmon|rockfish|halibut|fish|squid/.test(s)) return "fish";
  return "other";
}

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
        <div className="relative border-b border-hairline border-l-[3px] border-l-official-line bg-surface px-6 pb-4 pt-6">
          <p className="text-[13px] font-medium text-official">{OFFICIAL_STATUS.heading}</p>
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
        </div>
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-6 pb-8 pt-4">
          {verification && verification.state !== "verified" && ds && (
            // one line, the reasons one tap away: the list itself comes first
            <details className="rounded-lg border border-hairline border-l-[3px] border-l-official-line bg-surface px-3 py-2 text-[13px]" data-testid="drawer-why">
              <summary className="cursor-pointer text-official-ink">
                <b className="font-semibold">Transcribed from agency pages, not verified by a person.</b> Why?
              </summary>
              <div className="mt-2">
                <VerificationDetail v={verification} />
              </div>
            </details>
          )}
          {!ds && (
            <p className="text-[14px] text-ink-2">
              {OFFICIAL_STATUS.notTracked} {OFFICIAL_STATUS.instruction}
            </p>
          )}
          {disputedRecords(ds).length > 0 && (
            <section aria-labelledby="drawer-disputed" className="space-y-1.5" data-testid="drawer-disputed">
              <h3 id="drawer-disputed" className="text-[13px] font-medium text-warning">
                Reported lifted by the agency · awaiting review · {disputedRecords(ds).length}
              </h3>
              {disputedRecords(ds).map(({ record, dispute }) => (
                <NoticeCard key={record.id} r={record} now={now} dispute={dispute} compact />
              ))}
            </section>
          )}
          {GROUPS.map((g) => {
            const rs = records.filter((r) => groupOf(r) === g.id);
            if (!rs.length) return null;
            return (
              <section key={g.id} aria-labelledby={`drawer-${g.id}`} className="space-y-1.5" data-testid={`drawer-group-${g.id}`}>
                <h3 id={`drawer-${g.id}`} className="text-[13px] font-medium text-ink-3">
                  {g.label} · {rs.length}
                </h3>
                {rs.map((r) => (
                  <NoticeCard key={r.id} r={r} now={now} compact />
                ))}
              </section>
            );
          })}
          {ds && ds.registry.statements.length > 0 && (
            <details className="rounded-md border border-hairline px-3 py-2 text-[13px] text-ink-2">
              <summary className="cursor-pointer font-medium text-ink">{OFFICIAL_STATUS.statementsHeading}</summary>
              <p className="mt-1 text-[13px] text-ink-3">{OFFICIAL_STATUS.statementsNote}</p>
              <ul className="mt-1.5 space-y-1.5">
                {ds.registry.statements.map((s) => (
                  <li key={s.id}>
                    <p className="text-[13px] text-ink-3">
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
  { key: "coast", href: "/coast", label: "My Coast", icon: "pin" },
  { key: "bloom", href: "/bloom", label: "Blooms", icon: "bloom" },
  { key: "fisheries", href: "/fisheries", label: "Fisheries", icon: "fish" },
] as const;

/** Mobile bottom navigation: five places (signature P10). Notices is a tab, so official information is one tap from anywhere; About is in the masthead menu. */
export function TabBar({ active }: { active: string }) {
  const { ds, records, open, openDrawer } = useOfficial();
  const bar = useRef<HTMLElement>(null);
  return (
    <nav
      ref={bar}
      aria-label="Primary"
      data-testid="tabbar"
      className="fixed inset-x-0 bottom-0 z-40 grid h-[var(--cw-tabbar-h)] grid-cols-5 border-t border-hairline bg-surface pb-[env(safe-area-inset-bottom)] text-ink-3 md:hidden"
      style={{ colorScheme: "light" }}
    >
      {/* the active marker glides from the previous page's tab */}
      <Glide id="tabbar" container={bar} activeSelector='a[aria-current="page"]' className="pointer-events-none absolute top-0 h-0.5 rounded-b bg-accent" />
      {TABS.map((t) => (
        <Link
          key={t.key}
          href={t.href}
          data-testid={`tab-${t.key}`}
          aria-current={active === t.key ? "page" : undefined}
          className={`relative flex flex-col items-center justify-center gap-1 text-[13px] font-medium max-[379px]:text-[12px] ${active === t.key ? "text-accent" : ""} cw-press`}
        >
          <Icon name={t.icon} className="h-[22px] w-[22px]" />
          {t.key === "coast" ? (
            <span className="whitespace-nowrap">
              <span className="max-[379px]:hidden">My </span>Coast
            </span>
          ) : (
            t.label
          )}
        </Link>
      ))}
      <button
        type="button"
        data-testid="tab-notices"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={(e) => openDrawer(e.currentTarget)}
        className="relative flex flex-col items-center justify-center gap-1 text-[13px] font-medium text-official max-[379px]:text-[12px]"
      >
        <Icon name="shield" className="h-[22px] w-[22px]" />
        {ds && (
          <span className="absolute left-[calc(50%+6px)] top-2 min-w-[18px] rounded-full bg-official px-1.5 text-center text-[13px] font-semibold leading-[18px] text-white tabular">
            {records.length}
            <span className="sr-only"> active</span>
          </span>
        )}
        Notices
      </button>
    </nav>
  );
}
