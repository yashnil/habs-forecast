"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { FisheryId } from "@/pilot/contracts";
import { FISHERY_OPTIONS, dayWords, type StatusRow } from "@/pilot/facts";
import { Icon } from "@/v4/ui/Icon";
import { FactAnchors, Section, SourceLink, StatusMark, Unverified, factProps, sevClass } from "./ui";

/** The status strip: one row per fishery the reader follows, official status first. A row opens its detail sheet. */
export function StatusStrip({
  rows,
  now,
  place,
  statusError,
  otherNotices = [],
  onOpen,
  onEdit,
}: {
  rows: StatusRow[];
  now: string;
  place: string;
  statusError: boolean;
  /** notice facts tied to this port that belong to no row shown (their address is the section itself) */
  otherNotices?: string[];
  onOpen: (r: StatusRow) => void;
  onEdit: () => void;
}) {
  // a notice is addressed inside the first row that lists it (one id, once on the page)
  const placed = new Set<string>();
  const noticeIds = rows.map((r) => r.notices.map((n) => `notice:${n.id}`).filter((id) => !placed.has(id) && (placed.add(id), true)));
  return (
    <Section
      id="status"
      area="status"
      title="Your fisheries · official status"
      feedback={rows[0] ? { factId: rows[0].fact.id, placeId: place } : null}
      aside={
        <button type="button" onClick={onEdit} className="nx-focus nx-tint -my-2 inline-flex min-h-11 items-center gap-1 rounded-[3px] px-2 text-[14px] text-n-ink-2 hover:text-n-ink" data-testid="fisheries-edit">
          Edit list
        </button>
      }
    >
      {statusError && (
        <p className="today-say t-amber" role="status">
          Neritic&rsquo;s list of seasons and restrictions did not load. Unknown is not open: check{" "}
          <a className="nx-focus underline underline-offset-4" href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noreferrer">
            CDFW
          </a>{" "}
          and{" "}
          <a className="nx-focus underline underline-offset-4" href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noreferrer">
            CDPH
          </a>
          .
        </p>
      )}
      <ul className="mt-0" data-testid="status-strip">
        {rows.map((r, i) => (
          <li key={r.fishery} {...factProps(r.fact.id)}>
            <FactAnchors ids={noticeIds[i]} />
            <button type="button" className="today-row nx-focus nx-tint" onClick={() => onOpen(r)} data-testid={`status-${r.fishery}`} data-severity={r.severity} aria-haspopup="dialog">
              <StatusMark severity={r.severity} className="mt-0.5" />
              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="text-[16.5px] font-semibold leading-snug text-n-ink">{r.label}</span>
                  <Unverified verified={r.verified} />
                </span>
                <span className="mt-0.5 block text-[15px] leading-snug text-n-ink-2">{r.headline}</span>
                <span className="today-src mt-1 block">
                  {[r.causeWord ? `Cause: ${r.causeWord}` : null, r.next_review ? `Next review ${dayWords(r.next_review, now)}` : null, r.source ? `${r.source.name}${r.checked_at ? `, checked ${dayWords(r.checked_at, now)}` : ""}` : null].filter(Boolean).join(" · ")}
                  {r.stale && <span className="t-amber"> · check is old</span>}
                </span>
              </span>
              <span className="flex flex-col items-end gap-2">
                <span className={`today-word ${sevClass(r.severity)}`} data-testid="status-word">
                  {r.level === "no_restriction_on_record" ? "None on record" : r.word}
                </span>
                <Icon name="chevronRight" className="today-row-go h-4 w-4 text-n-ink-3" />
              </span>
            </button>
          </li>
        ))}
      </ul>
      <FactAnchors ids={otherNotices.filter((id) => !placed.has(id))} />
      <p className="nx-margin mt-3">
        Transcribed from CDFW and CDPH. <b>No restriction on record</b> is not a promise that a fishery is open or that seafood is free of toxin; notices can lag.
      </p>
    </Section>
  );
}

/** A native modal dialog (focus is trapped and returned by the browser), drawn as a bottom sheet on phones. */
export function Sheet({ open, onClose, labelledBy, children, testid }: { open: boolean; onClose: () => void; labelledBy: string; children: ReactNode; testid: string }) {
  const ref = useRef<HTMLDialogElement | null>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      className="today-sheet"
      aria-labelledby={labelledBy}
      onClose={onClose}
      onClick={(e) => {
        // a tap on the backdrop closes
        if (e.target === ref.current) onClose();
      }}
      data-testid={testid}
    >
      {open && children}
    </dialog>
  );
}

function Dl({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="mt-4 grid grid-cols-[minmax(0,8.5rem)_minmax(0,1fr)] gap-x-4 gap-y-2.5 text-[15px]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-n-ink-3">{k}</dt>
          <dd className="text-n-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function StatusSheet({ row, now, onClose }: { row: StatusRow | null; now: string; onClose: () => void }) {
  const r = row;
  return (
    <Sheet open={!!r} onClose={onClose} labelledBy="status-sheet-h" testid="status-sheet">
      {r && (
        <div className="flex max-h-[88dvh] flex-col md:max-h-dvh">
          <div className="flex items-start gap-3 border-b border-n-line px-4 py-4">
            <StatusMark severity={r.severity} className="mt-1" />
            <div className="min-w-0 flex-1">
              <p className={`today-word inline-block ${sevClass(r.severity)}`}>{r.word}</p>
              <h2 id="status-sheet-h" className="mt-2 font-serif text-[26px] leading-tight text-n-ink">
                {r.label}
              </h2>
            </div>
            <button type="button" onClick={onClose} className="nx-focus nx-press grid h-11 w-11 place-items-center rounded-[3px] border border-n-line" aria-label="Close" data-testid="sheet-close">
              <Icon name="close" />
            </button>
          </div>
          <div className="overflow-y-auto px-4 pb-8 pt-3">
            <p className="text-[17px] leading-snug text-n-ink">{r.headline}</p>
            {r.fact.detail && <p className="mt-2 text-[15px] leading-relaxed text-n-ink-2">{r.fact.detail}</p>}
            <Dl
              rows={[
                ...(r.causeWord ? ([["Cause", r.causeWord]] as [string, ReactNode][]) : []),
                ...(r.area ? ([["Area", r.area]] as [string, ReactNode][]) : []),
                ...(r.records[0]?.effective ? ([["Effective", dayWords(r.records[0].effective, now)]] as [string, ReactNode][]) : []),
                ...(r.records[0]?.until ? ([["Until", dayWords(r.records[0].until, now)]] as [string, ReactNode][]) : []),
                ...(r.next_review ? ([["Next review", dayWords(r.next_review, now)]] as [string, ReactNode][]) : []),
                ...(r.source ? ([["Source", <SourceLink key="s" name={r.source.name} url={r.source.url} />]] as [string, ReactNode][]) : []),
                ...(r.checked_at ? ([["Checked", dayWords(r.checked_at, now)]] as [string, ReactNode][]) : []),
                ["Verified", r.verified === false ? <span key="v" className="text-n-official">Not verified: no person has checked this record against the source yet</span> : r.verified ? "Checked by a person against the source" : "No curated record"],
              ]}
            />
            {r.records.length > 1 && (
              <>
                <h3 className="today-h2 mt-6">Also on record</h3>
                <ul className="mt-2">
                  {r.records.slice(1).map((x) => (
                    <li key={x.id} className="border-t border-n-line py-2 text-[14.5px] text-n-ink-2">
                      {x.headline}
                    </li>
                  ))}
                </ul>
              </>
            )}
            {r.notices.length > 0 && (
              <>
                <h3 className="today-h2 mt-6">Official notices for this fishery here</h3>
                <ul className="mt-2">
                  {r.notices.map((n) => (
                    <li key={n.id} className="border-t border-n-line py-2.5 text-[14.5px]">
                      <span className="text-n-ink">
                        {n.agency}: {n.title}
                      </span>{" "}
                      <Unverified verified={n.verified} />
                      {n.url && (
                        <a href={n.url} target="_blank" rel="noreferrer" className="nx-focus ml-1 inline-flex min-h-11 items-center gap-1 text-n-ink-2 underline underline-offset-4">
                          source <Icon name="external" className="h-3.5 w-3.5" />
                        </a>
                      )}
                    </li>
                  ))}
                </ul>
              </>
            )}
            <p className="nx-margin mt-6">Neritic is not an official source. CDFW and CDPH decide closures and reopenings; call before you go: CDFW closure line (831) 649-2883, CDPH biotoxin line 1 (800) 553-4133.</p>
          </div>
        </div>
      )}
    </Sheet>
  );
}

export function FisheryPicker({ open, selected, onClose, onSave }: { open: boolean; selected: FisheryId[]; onClose: () => void; onSave: (ids: FisheryId[]) => void }) {
  const [pick, setPick] = useState<FisheryId[]>(selected);
  useEffect(() => {
    if (open) setPick(selected);
  }, [open, selected]);
  return (
    <Sheet open={open} onClose={onClose} labelledBy="fisheries-sheet-h" testid="fisheries-sheet">
      <form
        className="flex max-h-[88dvh] flex-col md:max-h-dvh"
        onSubmit={(e) => {
          e.preventDefault();
          onSave(FISHERY_OPTIONS.map((o) => o.id).filter((id) => pick.includes(id)));
        }}
      >
        <div className="flex items-center justify-between border-b border-n-line px-4 py-3">
          <h2 id="fisheries-sheet-h" className="font-serif text-[24px] text-n-ink">
            Fisheries you follow
          </h2>
          <button type="button" onClick={onClose} className="nx-focus nx-press grid h-11 w-11 place-items-center rounded-[3px] border border-n-line" aria-label="Close">
            <Icon name="close" />
          </button>
        </div>
        <fieldset className="overflow-y-auto px-4 py-2">
          <legend className="sr-only">Fisheries shown in the status strip</legend>
          {FISHERY_OPTIONS.map((o) => (
            <label key={o.id} className="flex min-h-12 cursor-pointer items-center gap-3 border-b border-n-line text-[16px] text-n-ink">
              <input
                type="checkbox"
                className="h-5 w-5 accent-[var(--nx-accent)]"
                checked={pick.includes(o.id)}
                onChange={(e) => setPick((p) => (e.target.checked ? [...p, o.id] : p.filter((x) => x !== o.id)))}
                data-testid={`pick-${o.id}`}
              />
              {o.label}
            </label>
          ))}
          <p className="today-src mt-3">Kept on this device only.</p>
        </fieldset>
        <div className="flex gap-2 border-t border-n-line px-4 py-3">
          <button type="submit" disabled={!pick.length} className="today-btn primary nx-focus nx-press flex-1 disabled:opacity-50" data-testid="fisheries-save">
            Show these
          </button>
        </div>
      </form>
    </Sheet>
  );
}
