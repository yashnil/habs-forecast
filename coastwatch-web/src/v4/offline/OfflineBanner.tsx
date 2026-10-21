"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel } from "@/v4/ui/time";
import type { PackProblem, PackRecord } from "./packs";
import { ageWords } from "./snapshot";
import { packAge } from "./useOfflinePack";

/**
 * Offline: unmistakable and persistent, with the time of what is shown and, once it is outdated, how old
 * it is; one line until opened. Never a toast. Shown only offline (a client-only state), so its
 * device-time words never meet server HTML. `outside`: the place on screen is not in any saved pack
 * (its name, and where the saved pack's own page is), so the banner never claims the pack covers it.
 * The pack saves the briefing for all species (for=all&species=any): a briefing for another trip says
 * so ("saved for all species"), never that it was saved for the reader's own trip.
 * On Explore at phone width it folds into the status chip (StatusCenter), which then says "Offline"
 * and holds the same words: the map keeps its height.
 */
export function OfflineBanner({ record, name, problem = null, inline = false, outside = null }: { record: PackRecord | null; name: string; problem?: PackProblem | null; inline?: boolean; outside?: { place: string; href: string } | null }) {
  const [open, setOpen] = useState(false);
  // read after mount (this banner is client-only state): the page, and whether its trip is the saved one
  const [where, setWhere] = useState<{ explore: boolean; otherTrip: boolean }>({ explore: false, otherTrip: false });
  useEffect(() => {
    const u = new URL(window.location.href);
    const f = u.searchParams.get("for");
    const sp = u.searchParams.get("species");
    setWhere({ explore: u.pathname === "/", otherTrip: u.pathname === "/coast" && ((!!f && f !== "all") || (!!sp && sp !== "any")) });
  }, []);
  const older = problem === "older-format";
  const pa = record?.complete ? packAge(record, new Date()) : null;
  const age = pa ? ageWords({ ageHours: pa.hours, outdated: pa.outdated, savedAt: record?.savedAt }) : null;
  const forAll = where.otherTrip && record?.complete && !outside ? " · saved for all species" : "";
  const short = outside
    ? `Offline · ${outside.place} is not saved for offline use`
    : record?.complete
    ? `Offline · ${name} saved ${clockLabel(record.savedAt)}${age ? ` · ${age}` : ""}${older ? " (older pack)" : ""}${forAll}`
    : record || problem === "unfinished" || problem === "damaged-record"
      ? `Offline · ${name} pack incomplete`
      : "Offline · no field pack saved";
  return (
    <div role="status" className={`${inline ? "relative" : "env-chart absolute inset-x-3 top-[64px] z-40 max-md:top-[118px]"} ${where.explore ? "max-sm:hidden" : ""} mx-auto max-w-[640px] rounded-2xl border border-n-line-strong bg-n-surface text-n-ink shadow-[var(--nx-shadow)]`} data-testid="offline-banner">
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="nx-focus flex min-h-12 w-full items-center gap-3 px-4 text-left">
        <Icon name="offline" className="h-5 w-5 shrink-0 text-n-offline" />
        <span className="min-w-0 flex-1 text-[14.5px] font-semibold leading-snug">{short}</span>
        <Icon name="chevronDown" className={`h-5 w-5 shrink-0 text-n-ink-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {outside && (
        <p className="px-4 pb-3 text-[14px] leading-snug text-n-ink-2" data-testid="offline-outside">
          Nothing for {outside.place} is saved on this device: its evidence needs a connection.{" "}
          {record?.complete ? (
            <Link href={outside.href} className="nx-focus font-semibold text-n-accent underline underline-offset-2">
              Open {name}, saved {clockLabel(record.savedAt)}
            </Link>
          ) : (
            "No field pack is saved on this device."
          )}
        </p>
      )}
      {open && !outside && (
        <p className="px-4 pb-3 text-[14px] leading-snug text-n-ink-2">
          {record?.complete
            ? `${forAll ? "The saved briefing is for all species, not only your trip's; your species' own lines need a connection. " : ""}Every value keeps its own time; nothing here is live. Official notices as last checked${record.noticesCheckedAt ? ` ${clockLabel(record.noticesCheckedAt)}` : ""}: confirm with CDFW and CDPH when you are back online.${older ? " Saved by an earlier version of Neritic: some parts may not open; download again when you have a connection." : ""}`
            : record || problem
              ? "Some parts of the pack were not saved and may not open. Finish the download when you have a connection."
              : "Only pages already open keep working. Save a field pack while you have a connection."}
        </p>
      )}
    </div>
  );
}
