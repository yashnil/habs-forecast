"use client";

import { useId, useState } from "react";
import type { Manifest } from "@/generated/schema";
import { isFixture } from "@/lib/layers";
import { formatDateTimePT } from "@/lib/time";
import { Icon } from "@/components/ui/Icon";
import { CDFW_URL, CDPH_URL, useRegistryDisclosure } from "@/components/shell/OfficialShell";
import { useMode } from "@/lib/mode";

const link = "font-semibold text-white underline decoration-white/40 underline-offset-2 hover:decoration-white";

/**
 * The status line, inside the navy masthead (signature handoff #1): the unverified-notices
 * disclosure, failed data updates, test data and missing data share one row with a 3 px amber
 * left rule, instead of a second cream band. The row always carries the essential words and the
 * agency links; "More" opens the reasons. Missing live data is an alert and leads the row.
 */
export function DataBanners({ manifest, error }: { manifest: Manifest | null; error: string | null }) {
  const registry = useRegistryDisclosure();
  // Both modes keep the standing line while the notices are unreviewed, so every page links straight
  // to CDFW and CDPH (the agencies are the authority; the public research preview must say so on
  // every page, not only behind the notices pill). Failed updates stay Research-only here: Coastal
  // shows them in the data-status dot and on each layer that is affected; missing data and test
  // data always show.
  const [mode] = useMode();
  const reg = registry;
  const [open, setOpen] = useState(false);
  const id = useId();
  const unavailable = !!error || !manifest;
  // Coastal shows failed updates only in the data-status dot and on the layers affected
  const failed = manifest && mode === "research" ? manifest.sources.filter((s) => s.outcome === "failed") : [];
  const fixture = !!manifest && isFixture(manifest);
  if (!reg.show && !unavailable && !failed.length && !fixture) {
    // same box as the line below, with an invisible line of text: no jump when the line appears
    return reg.pending ? (
      <div aria-hidden className="relative z-20 border-t border-white/[0.06] text-[13px] leading-snug" data-testid="status-line-pending">
        {/* the line's one-row height: its "More" button sets it */}
        <div className="mx-auto flex min-h-[33.875px] max-w-[1600px] items-center px-4 py-1">
          <span className="invisible">Official notices</span>
        </div>
      </div>
    ) : null;
  }

  const failedNames = failed.map((s) => s.title.split(":")[0].replace(/ harmful algal bloom forecast$/, ""));
  const hasMore = reg.show || failed.length > 0 || fixture;

  return (
    <div className="relative z-20 border-t border-white/[0.06] text-[13px] leading-snug text-on-navy" data-testid="status-line">
      <div className="flex min-h-8 items-center gap-x-3 border-l-[3px] border-official-on-navy py-1 pl-[13px] pr-2 md:pl-[21px] md:pr-4">
        <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-3 gap-y-0.5 md:flex-nowrap">
          {unavailable && (
            <p role="alert" data-testid="data-unavailable" className="min-w-0 text-on-navy">
              <strong className="font-semibold text-official-on-navy">Live data unavailable.</strong> Forecast and satellite layers cannot be shown{error ? ` (${error})` : ""}. Official closures and advisories are always on the CDFW and CDPH sites.
            </p>
          )}
          {reg.show && (
            <p role="status" data-testid="registry-disclosure" className="flex min-w-0 items-baseline gap-1.5">
              <Icon name="shield" className="h-3.5 w-3.5 shrink-0 translate-y-[2px] text-official-on-navy max-sm:hidden" />
              {/* phones: one short line; the drawer holds the detail */}
              <span className="sm:hidden">
                <b className="font-semibold text-official-on-navy">Notices not verified.</b> Check{" "}
                <a href={CDFW_URL} target="_blank" rel="noopener noreferrer" className={link}>CDFW</a> and{" "}
                <a href={CDPH_URL} target="_blank" rel="noopener noreferrer" className={link}>CDPH</a>.
              </span>
              {/* tablets: short enough that "See the list" is never truncated away */}
              <span className="hidden sm:inline lg:hidden">
                <b className="font-semibold text-official-on-navy">Official notices are not verified.</b> Check{" "}
                <a href={CDFW_URL} target="_blank" rel="noopener noreferrer" className={link}>CDFW</a> and{" "}
                <a href={CDPH_URL} target="_blank" rel="noopener noreferrer" className={link}>CDPH</a>.{" "}
                <button type="button" onClick={(e) => reg.openDrawer(e.currentTarget)} className={link}>
                  See the list
                </button>
              </span>
              <span className="max-lg:hidden lg:truncate">
                <b className="font-semibold text-official-on-navy">Official notices are not verified.</b> The list may be incomplete. Before harvesting or eating seafood, check{" "}
                <a href={CDFW_URL} target="_blank" rel="noopener noreferrer" className={link}>CDFW</a> and{" "}
                <a href={CDPH_URL} target="_blank" rel="noopener noreferrer" className={link}>CDPH</a>.{" "}
                <button type="button" onClick={(e) => reg.openDrawer(e.currentTarget)} className={link}>
                  See the list
                </button>
              </span>
              <span className="sr-only">{reg.reason}</span>
            </p>
          )}
          {failed.length > 0 && mode === "research" && (
            <p role="status" data-testid="source-failure-banner" className="flex shrink-0 items-baseline gap-1.5 text-on-navy-2 max-md:hidden">
              <span aria-hidden className="text-on-navy-2">●</span>
              <span className="sr-only">Latest update failed for {failed.map((s) => s.title).join(", ")}.</span>
              <span aria-hidden>Latest update failed: {failedNames.join(", ")}</span>
            </p>
          )}
          {fixture && (
            <p role="status" data-testid="fixture-banner" className="shrink-0 rounded-full border border-official-on-navy/50 px-2 text-[13px] font-semibold text-official-on-navy">
              Test data<span className="sr-only">: recorded fixture data, not a live feed</span>
            </p>
          )}
        </div>
        {hasMore && (
          <button
            type="button"
            aria-expanded={open}
            aria-controls={id}
            onClick={() => setOpen(!open)}
            data-testid="status-more"
            className="cw-press ml-auto flex shrink-0 items-center gap-1 rounded-md px-1.5 py-1 text-[13px] font-semibold text-on-navy hover:bg-white/[0.07]"
          >
            {/* failed updates are counted in Research; Coastal keeps them under "More" (and on each layer) */}
            {failed.length > 0 && mode === "research" && (
              <span className="rounded-full bg-white/10 px-1.5 text-[13px] text-on-navy md:hidden" aria-hidden>
                +{failed.length}
              </span>
            )}
            <span className="max-[400px]:sr-only">{open ? "Less" : "More"}</span>
            <svg viewBox="0 0 12 12" className={`h-3 w-3 transition-transform duration-200 ${open ? "rotate-180" : ""}`} aria-hidden>
              <path d="M3 4.5 6 7.5l3-3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
      {open && (
        <div id={id} className="theme-paper absolute inset-x-0 top-full border-b border-hairline bg-surface shadow-[0_8px_20px_rgba(6,17,30,0.18)]">
          <div className="mx-auto max-w-[1600px] space-y-1.5 border-l-[3px] border-official-line px-4 py-2.5 text-[13px] text-ink-2 md:px-6">
            {reg.show && (
              <p>
                <b className="font-semibold text-official-ink">Why the notices are not verified.</b> {reg.reason}{" "}
                <button type="button" onClick={(e) => reg.openDrawer(e.currentTarget)} className="font-semibold text-official-ink underline underline-offset-2">
                  See the list
                </button>
              </p>
            )}
            {failed.length > 0 && (
              <p>
                <b className="font-semibold text-ink">Latest update failed</b> for {failed.map((s) => s.title).join("; ")}. The last successful data is shown with its real dates.{" "}
                <a href="/sources" className="font-semibold text-ink underline underline-offset-2">
                  Data status
                </a>
              </p>
            )}
            {fixture && (
              <p>
                <b className="font-semibold text-ink">Test data.</b> This build shows recorded fixture data (C-HARM subset for Monterey Bay and the Gulf of the Farallones, recorded {formatDateTimePT("2026-10-08T17:02:00Z")}), not a live feed.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
