"use client";

import { OFFICIAL_STATUS } from "@/content/copy";
import { useOfficialDrawer } from "@/components/shell/OfficialShell";
import { Icon } from "@/components/ui/Icon";

/**
 * Desktop place chip (M5): one line at the top left of the map with the place being viewed
 * (opens the place list) and the official notices for it with their verification state
 * (opens the notices list). Replaces the tall place card so the coast stays visible; the
 * notice count and "not verified" never leave the screen.
 */
export function PlaceChip({
  label,
  open,
  onToggle,
  regionNotices,
  statewide,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
  /** notices that may apply in the region on view; null statewide */
  regionNotices: number | null;
  statewide: boolean;
}) {
  const { openDrawer, count, verification, available } = useOfficialDrawer();
  const n = !statewide && regionNotices != null ? regionNotices : count;
  const ver = verification ? OFFICIAL_STATUS.verification[verification.state] : "Checking…";
  const unverified = verification?.state !== "verified";
  return (
    <div data-testid="place-chip" className="theme-paper flex h-10 items-stretch overflow-hidden rounded-xl bg-surface text-ink shadow-[0_1px_2px_rgba(6,17,30,0.14),0_6px_18px_rgba(6,17,30,0.24)]">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        aria-controls="places-panel"
        data-testid="place-chip-toggle"
        className="flex min-w-0 items-center gap-2 pl-3 pr-2.5 text-[14px] font-semibold hover:bg-surface-2"
      >
        <Icon name="pin" className="h-4 w-4 shrink-0 text-accent" />
        <span className="truncate">{label}</span>
        <Icon name="chevron" className={`h-3.5 w-3.5 shrink-0 text-ink-3 transition-transform duration-150 ${open ? "-rotate-90" : "rotate-90"}`} />
      </button>
      <button
        type="button"
        data-testid="official-summary"
        onClick={(e) => openDrawer(e.currentTarget)}
        aria-haspopup="dialog"
        className="flex items-center gap-1.5 border-l border-official-line bg-official-bg px-3 text-[13px] text-official-ink hover:brightness-[0.97]"
      >
        <Icon name="shield" className="h-4 w-4 shrink-0 text-official" />
        {available ? (
          <>
            <b className="font-semibold tabular">
              {n} notice{n === 1 ? "" : "s"}
            </b>
            <span className="sr-only">{!statewide && regionNotices != null ? ` may apply in ${label}` : " active in California"}, </span>
            <span className={`whitespace-nowrap ${unverified ? "font-semibold" : ""}`}>· {ver}</span>
          </>
        ) : (
          <b className="font-semibold">Notices unavailable: check CDFW and CDPH</b>
        )}
      </button>
    </div>
  );
}
