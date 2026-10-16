"use client";

import { useEffect, useRef, useState } from "react";
import type { OfficialDataset } from "@/generated/official";
import { currentRecords, disputedRecords } from "@/lib/official";
import { Icon } from "@/v4/ui/Icon";
import { dayLabel } from "@/v4/ui/time";

/**
 * Official notices: one control, always saying they are not verified until a person reviews the
 * registry (cwp review-official). Each notice links to its agency's own page; the agencies' lists
 * are the authority, Neritic's copy is a transcription.
 */
export function NoticesButton({ official, compact = false }: { official: OfficialDataset | null; compact?: boolean }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);
  const recs = official ? currentRecords(official) : [];
  // an agency release appears to lift these: never counted or shown as in effect, always shown with the release
  const disputed = official ? disputedRecords(official) : [];
  const reviewed = (official as unknown as { registry?: { review?: { status?: string } } } | null)?.registry?.review?.status === "human_verified";
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
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
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label={`${recs.length} official notices, ${reviewed ? "verified" : "not verified"}`}
        data-testid="notices-button"
        data-state={official ? (reviewed ? "verified" : "unverified") : "unavailable"}
        className="nx-press nx-focus flex h-11 items-center gap-2 rounded-full border border-n-official-line bg-n-surface px-3 text-[14px] font-semibold text-n-ink"
      >
        <Icon name="shield" className="h-[18px] w-[18px] text-n-official" />
        <span className="nx-data">{official ? recs.length : "?"}</span>
        {!compact && <span className="text-n-official">{reviewed ? "Verified" : "Not verified"}</span>}
      </button>
      {open && (
        <div role="dialog" aria-label="Official notices" className="nx-rise absolute right-0 top-[52px] z-50 flex max-h-[min(70vh,640px)] w-[min(420px,calc(100vw-24px))] flex-col rounded-2xl border border-n-line bg-n-surface text-n-ink shadow-[var(--nx-shadow)]" data-testid="notices-panel">
          <div className="border-b border-n-line p-4">
            <p className="text-[16px] font-semibold" data-testid="notices-heading">
              {recs.length} active notices in California
            </p>
            <p className="mt-1 text-[13.5px] leading-snug text-n-official" data-testid="notices-unverified">
              {reviewed ? "Checked by a person against the agencies' pages." : "Not verified. Neritic's copy has not been checked by a person; the list may be incomplete or out of date."} Before harvesting or eating seafood, check CDFW and CDPH.
            </p>
            <div className="mt-2 flex gap-2">
              <a href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold">
                CDFW <Icon name="external" className="h-3.5 w-3.5" />
              </a>
              <a href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold">
                CDPH <Icon name="external" className="h-3.5 w-3.5" />
              </a>
            </div>
          </div>
          <div className="flex-1 overflow-auto">
          {disputed.length > 0 && (
            <section className="border-b border-n-line px-4 py-3" data-testid="notices-disputed">
              <h3 className="text-[13.5px] font-semibold text-n-stale">Reported lifted by the agency · awaiting review · {disputed.length}</h3>
              <ul className="mt-1">
                {disputed.map(({ record: r, dispute: d }) => (
                  <li key={r.id} className="py-2" data-testid={`notice-${r.id}`} data-disputed="true">
                    <p className="text-[14.5px] font-semibold leading-snug">{r.title}</p>
                    <p className="mt-0.5 text-[13px] text-n-ink-2">
                      {d.agency} release {d.release} (&ldquo;{d.title}&rdquo;) appears to lift this notice. Not reviewed by a person yet, so it is not counted as in effect.
                    </p>
                    <div className="flex flex-wrap gap-x-3">
                      <a href={d.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-9 items-center gap-1 text-[13px] font-semibold text-n-accent">
                        {d.agency} {d.release} <Icon name="external" className="h-3.5 w-3.5" />
                      </a>
                      {r.sources[0] && (
                        <a href={r.sources[0].url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-9 items-center gap-1 text-[13px] font-semibold text-n-accent">
                          The original notice <Icon name="external" className="h-3.5 w-3.5" />
                        </a>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}
          <ul className="divide-y divide-n-line px-4" data-testid="notices-current">
            {recs.map((r) => (
              <li key={r.id} className="py-3" data-testid={`notice-${r.id}`}>
                <p className="text-[14.5px] font-semibold leading-snug">{r.title}</p>
                <p className="mt-0.5 text-[13px] text-n-ink-2">
                  {r.agency}
                  {r.effective_date ? ` · since ${dayLabel(r.effective_date)}` : ""}
                </p>
                {r.sources[0] && (
                  <a href={r.sources[0].url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-9 items-center gap-1 text-[13px] font-semibold text-n-accent">
                    {r.agency} source <Icon name="external" className="h-3.5 w-3.5" />
                  </a>
                )}
              </li>
            ))}
            {!official && <li className="py-3 text-[14px] text-n-ink-2">Neritic&rsquo;s copy of the notices could not be loaded. Check CDFW and CDPH directly.</li>}
          </ul>
          </div>
        </div>
      )}
    </div>
  );
}
