"use client";

import { useState } from "react";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel } from "@/v4/ui/time";
import type { PackRecord } from "./packs";

/** Offline: unmistakable and persistent, with the time of what is shown; one line until opened. Never a toast. */
export function OfflineBanner({ record, name }: { record: PackRecord | null; name: string }) {
  const [open, setOpen] = useState(false);
  const short = record?.complete ? `Offline · ${name} saved ${clockLabel(record.savedAt)}` : record ? `Offline · ${name} pack incomplete` : "Offline · no field pack saved";
  return (
    <div role="status" className="env-chart absolute inset-x-3 top-[64px] z-40 mx-auto max-w-[640px] rounded-2xl border border-n-line-strong bg-n-surface text-n-ink shadow-[var(--nx-shadow)] max-md:top-[118px]" data-testid="offline-banner">
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="nx-focus flex min-h-12 w-full items-center gap-3 px-4 text-left">
        <Icon name="offline" className="h-5 w-5 shrink-0 text-n-offline" />
        <span className="min-w-0 flex-1 text-[14.5px] font-semibold leading-snug">{short}</span>
        <Icon name="chevronDown" className={`h-5 w-5 shrink-0 text-n-ink-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <p className="px-4 pb-3 text-[14px] leading-snug text-n-ink-2">
          {record?.complete
            ? `Every value keeps its own time; nothing here is live. Official notices as last checked${record.noticesCheckedAt ? ` ${clockLabel(record.noticesCheckedAt)}` : ""}: confirm with CDFW and CDPH when you are back online.`
            : record
              ? "Some parts of the pack were not saved and may not open. Finish the download when you have a connection."
              : "Only pages already open keep working. Save a field pack while you have a connection."}
        </p>
      )}
    </div>
  );
}
