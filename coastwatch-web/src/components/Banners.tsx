"use client";

import { useId, useState } from "react";
import type { Manifest } from "@/generated/schema";
import { isFixture } from "@/lib/layers";
import { formatDateTimePT } from "@/lib/time";
import { Icon } from "@/components/ui/Icon";
import { CDFW_URL, CDPH_URL, useRegistryDisclosure } from "@/components/shell/OfficialShell";

const link = "font-semibold underline underline-offset-2";

/**
 * One status line under the masthead (M5): the unverified-notices disclosure, failed data
 * updates, test data and missing data share a single row instead of stacking banners. The
 * row always carries the essential words and the agency links; "More" opens the reasons.
 * Missing live data is an alert and leads the row.
 */
export function DataBanners({ manifest, error }: { manifest: Manifest | null; error: string | null }) {
  const reg = useRegistryDisclosure();
  const [open, setOpen] = useState(false);
  const id = useId();
  const unavailable = !!error || !manifest;
  const failed = manifest ? manifest.sources.filter((s) => s.outcome === "failed") : [];
  const fixture = !!manifest && isFixture(manifest);
  if (!reg.show && !unavailable && !failed.length && !fixture) return null;

  const failedNames = failed.map((s) => s.title.split(":")[0].replace(/ harmful algal bloom forecast$/, ""));
  const hasMore = reg.show || failed.length > 0 || fixture;

  return (
    <div className="relative z-20 border-b border-official-line bg-official-bg text-[12.5px] leading-snug text-official-ink" data-testid="status-line">
      <div className="mx-auto flex min-h-8 max-w-[1600px] items-center gap-x-3 px-4 py-1">
        <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-3 gap-y-0.5 md:flex-nowrap">
          {unavailable && (
            <p role="alert" data-testid="data-unavailable" className="min-w-0 text-ink">
              <strong className="font-semibold text-serious">Live data unavailable.</strong> Forecast and satellite layers cannot be shown{error ? ` (${error})` : ""}. Official closures and advisories are always on the CDFW and CDPH sites.
            </p>
          )}
          {reg.show && (
            <p role="status" data-testid="registry-disclosure" className="flex min-w-0 items-baseline gap-1.5">
              <Icon name="shield" className="h-3.5 w-3.5 shrink-0 translate-y-[2px] text-official" />
              {/* phones: one short line; the drawer holds the detail */}
              <span className="sm:hidden">
                <b className="font-semibold">Notices not verified.</b> Check{" "}
                <a href={CDFW_URL} target="_blank" rel="noopener noreferrer" className={link}>CDFW</a> and{" "}
                <a href={CDPH_URL} target="_blank" rel="noopener noreferrer" className={link}>CDPH</a>.
              </span>
              <span className="max-sm:hidden md:truncate">
                <b className="font-semibold">Official notices are not verified.</b> The list may be incomplete. Before harvesting or eating seafood, check{" "}
                <a href={CDFW_URL} target="_blank" rel="noopener noreferrer" className={link}>CDFW</a> and{" "}
                <a href={CDPH_URL} target="_blank" rel="noopener noreferrer" className={link}>CDPH</a>.{" "}
                <button type="button" onClick={(e) => reg.openDrawer(e.currentTarget)} className={link}>
                  See the list
                </button>
              </span>
              <span className="sr-only">{reg.reason}</span>
            </p>
          )}
          {failed.length > 0 && (
            <p role="status" data-testid="source-failure-banner" className="flex shrink-0 items-baseline gap-1.5 text-ink max-md:hidden">
              <span aria-hidden className="text-warning">●</span>
              <span className="sr-only">Latest update failed for {failed.map((s) => s.title).join(", ")}.</span>
              <span aria-hidden>Latest update failed: {failedNames.join(", ")}</span>
            </p>
          )}
          {fixture && (
            <p role="status" data-testid="fixture-banner" className="shrink-0 rounded-full border border-warning/50 px-2 text-[11.5px] font-semibold text-warning">
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
            className="flex shrink-0 items-center gap-1 rounded-md px-1.5 py-1 text-[12px] font-semibold hover:bg-official-line/20"
          >
            {failed.length > 0 && (
              <span className="rounded-full bg-warning/15 px-1.5 text-[11px] text-warning md:hidden" aria-hidden>
                +{failed.length}
              </span>
            )}
            {open ? "Less" : "More"}
            <svg viewBox="0 0 12 12" className={`h-3 w-3 transition-transform duration-200 ${open ? "rotate-180" : ""}`} aria-hidden>
              <path d="M3 4.5 6 7.5l3-3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
      {open && (
        <div id={id} className="absolute inset-x-0 top-full border-b border-official-line bg-official-bg shadow-[0_8px_20px_rgba(6,17,30,0.18)]">
          <div className="mx-auto max-w-[1600px] space-y-1.5 px-4 py-2.5 text-ink-2">
            {reg.show && (
              <p>
                <b className="font-semibold text-official-ink">Why the notices are not verified.</b> {reg.reason}{" "}
                <button type="button" onClick={(e) => reg.openDrawer(e.currentTarget)} className={`${link} text-official-ink`}>
                  See the list
                </button>
              </p>
            )}
            {failed.length > 0 && (
              <p>
                <b className="font-semibold text-ink">Latest update failed</b> for {failed.map((s) => s.title).join("; ")}. The last successful data is shown with its real dates.{" "}
                <a href="/sources" className={`${link} text-ink`}>
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
