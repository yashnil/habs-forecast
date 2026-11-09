"use client";

import { useEffect, useState, type ReactNode } from "react";
import type { FisheryId } from "@/pilot/contracts";
import { FISHERY_OPTIONS, dayWords, rowHeadline, rowWord, unverifiedCount, type StatusRow, type WhaleView } from "@/pilot/facts";
import { Fold, Sheet, useFocusLink } from "@/v4/motion";
import { Icon } from "@/v4/ui/Icon";
import { Tag, type TagLevel } from "@/v4/ui/Tag";
import { Unverified } from "@/v4/ui/Unverified";
import { Button } from "@/v4/ui/Button";
import { Chapter, FactAnchors, NotVerifiedWords, SourceLink, StatusMark, factProps } from "./ui";
import { ZoneLadder } from "./Charts";

const tagLevel = (r: StatusRow): TagLevel => (r.conflict ? "watch" : r.level === "no_restriction_on_record" ? "none" : r.level === "closed" ? "closed" : r.level === "delayed" ? "delayed" : r.level === "restricted" ? "restricted" : "watch");

/** The word and its shape: a tag, never colour alone. A season that simply ended is said calmly, a barred disc in ink, not the alarm. */
function RowTag({ r, testId = "status-word" }: { r: StatusRow; testId?: string }) {
  if (r.seasonal)
    return (
      <span className="nx-tag" data-level="season" data-testid={testId}>
        <StatusMark severity={r.severity} seasonal className="!h-3.5 !w-3.5" />
        {rowWord(r)}
      </span>
    );
  return (
    <Tag level={tagLevel(r)} testId={testId}>
      {r.level === "no_restriction_on_record" && !r.conflict ? "None on record" : rowWord(r)}
    </Tag>
  );
}

/** "Rock crab" → "rock crab"; a proper name keeps its capital. */
const inList = (label: string) => (/^Dungeness/.test(label) ? label : label.charAt(0).toLowerCase() + label.slice(1));
const listWords = (xs: string[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`);

/** What a row or glance item lights on the stage: the notice's line, when it has one. */
export const statusFocusId = (r: StatusRow) => r.notices[0]?.id ?? r.records[0]?.id ?? null;

/**
 * Fisheries, chapter one: official status first. Flagged fisheries as large rows, each opening its sheet;
 * the fisheries with nothing on record collapse to one line that carries its own caveat; the whale rules
 * are one line beside them. The group's "not verified" marker is computed from its records.
 */
export function StatusStrip({
  rows,
  statusError,
  whale,
  otherNotices = [],
  onOpen,
  onOpenWhale,
  onEdit,
}: {
  rows: StatusRow[];
  statusError: boolean;
  whale: WhaleView | null;
  /** notice facts tied to this port that belong to no row shown (their address is the section itself) */
  otherNotices?: string[];
  onOpen: (r: StatusRow) => void;
  onOpenWhale: () => void;
  onEdit: () => void;
}) {
  const { bind } = useFocusLink();
  // a notice is addressed inside the first row that lists it (one id, once on the page)
  const placed = new Set<string>();
  const noticeIds = new Map(rows.map((r) => [r, r.level === "unknown" ? [] : r.notices.map((n) => `notice:${n.id}`).filter((id) => !placed.has(id) && (placed.add(id), true))]));
  const known = rows.filter((r) => r.level !== "unknown");
  const unknown = rows.filter((r) => r.level === "unknown");
  const flagged = known.filter((r) => r.level !== "no_restriction_on_record");
  const none = known.filter((r) => r.level === "no_restriction_on_record");
  const count = unverifiedCount([...known.map((r) => ({ verified: r.verified, known: true })), ...(whale ? [{ verified: whale.verified, known: true }] : [])]);
  const mark = (v: boolean | null) => (count.mixed && v !== true ? <NotVerifiedWords /> : null);
  const say = statusError
    ? null
    : flagged.length
      ? `${flagged.length} of your ${rows.length} ${rows.length === 1 ? "fishery is" : "fisheries are"} flagged.`
      : "Nothing is flagged for the fisheries you follow.";

  const row = (r: StatusRow) => (
    <li key={r.fishery} {...factProps(r.fact.id)} {...(statusFocusId(r) ? bind(statusFocusId(r)!) : {})}>
      <FactAnchors ids={noticeIds.get(r) ?? []} />
      <button type="button" className="today-row nx-focus nx-hover-tint" onClick={() => onOpen(r)} data-testid={`status-${r.fishery}`} data-severity={r.severity} data-seasonal={r.seasonal ? "" : undefined} aria-haspopup="dialog">
        <span className="today-row-name">{r.label}</span>
        <RowTag r={r} />
        <span className="today-row-body">
          {r.conflict ? <span data-testid="sources-differ">{rowHeadline(r)}</span> : r.headline}
          {r.level === "no_restriction_on_record" ? " Not a promise it is open." : ""}
          {r.stale ? <span className="today-old"> Check is old.</span> : null}
          {mark(r.verified) ? <> {mark(r.verified)}</> : null}
        </span>
        <Icon name="chevronRight" className="today-row-go h-4 w-4" />
      </button>
    </li>
  );

  return (
    <Chapter
      id="status"
      area="status"
      title="Fisheries"
      testid="today-status"
      say={say}
      marker={count.words ? <Unverified count={count.words} testId="status-not-verified" /> : undefined}
      action={
        <button type="button" onClick={onEdit} className="nx-focus today-edit" data-testid="fisheries-edit">
          Edit list
        </button>
      }
    >
      {statusError && (
        <p className="today-say" role="status">
          <span className="today-old">Neritic&rsquo;s list of seasons and restrictions did not load. Unknown is not open:</span> check{" "}
          <a className="nx-link nx-focus" href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noreferrer">
            CDFW
          </a>{" "}
          and{" "}
          <a className="nx-link nx-focus" href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noreferrer">
            CDPH
          </a>
          .
        </p>
      )}
      <div data-testid="status-strip">
        {(flagged.length > 0 || unknown.length > 0) && (
          <ul>
            {flagged.map(row)}
            {unknown.length > 0 && (
              // Neritic's list did not load: one statement for the fisheries it cannot speak to, never a row each that reads like a status
              <li className="today-row today-row-static" data-testid="status-unknown" data-severity="info" data-fact-card="">
                <span className="today-row-name">
                  <FactAnchors ids={unknown.map((r) => r.fact.id)} />
                  {unknown.length === rows.length ? "Your fisheries" : `${unknown.length} other ${unknown.length === 1 ? "fishery" : "fisheries"}`}
                </span>
                <span className="nx-tag" data-level="none" data-testid="status-word">
                  {unknown[0].word}
                </span>
                <span className="today-row-body">
                  {unknown.map((r) => r.label).join(" · ")}
                  <span className="block">Unknown is not open. CDFW closure line (831) 649-2883.</span>
                </span>
              </li>
            )}
          </ul>
        )}
        {none.length > 0 && (
          <Fold
            className="today-none"
            testid="status-none"
            title={
              <span data-testid="status-none-line">
                None on record for {listWords(none.map((r) => inList(r.label)))}. <b className="font-medium text-n-ink">Not a promise {none.length === 1 ? "it is" : "they are"} open.</b>
              </span>
            }
            bodyClassName="pt-1"
          >
            <ul>{none.map(row)}</ul>
          </Fold>
        )}
      </div>
      {whale ? (
        <div className="today-whale" {...factProps(whale.fact.id)}>
          <button type="button" className="today-row nx-focus nx-hover-tint" onClick={onOpenWhale} data-testid="status-whale" aria-haspopup="dialog">
            <span className="today-row-name">Whale rules</span>
            <span className="nx-caps">Zone {whale.zone}</span>
            <span className="today-row-body" data-testid="whale-lede">
              Zone {whale.zone}, {whale.label}: {(whale.action ?? "no action on record").replace(/\.+$/, "")}.{mark(whale.verified) ? <> {mark(whale.verified)}</> : null}
            </span>
            <Icon name="chevronRight" className="today-row-go h-4 w-4" />
          </button>
        </div>
      ) : (
        <p className="today-note" data-testid="whale-unavailable">
          Neritic has no current whale-entanglement (RAMP) record. See{" "}
          <a className="nx-link nx-focus" href="https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries" target="_blank" rel="noreferrer">
            CDFW Whale Safe Fisheries
          </a>
          .
        </p>
      )}
      <FactAnchors ids={otherNotices.filter((id) => !placed.has(id))} />
      <p className="nx-margin mt-4">Transcribed from CDFW and CDPH, whose notices take precedence. Notices can lag; Neritic checks them once a day.</p>
    </Chapter>
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

/** The RAMP zone, its rule and its next assessment: the whale row's sheet, and inside a crab fishery's sheet. */
function WhaleDetail({ whale, now }: { whale: WhaleView; now: string }) {
  return (
    <div className="mt-2 flex gap-4" data-testid="whale-detail">
      <ZoneLadder zone={whale.zone} severity={whale.fact.severity} />
      <div className="min-w-0 flex-1">
        <p className="text-[15px] leading-snug text-n-ink">
          Zone {whale.zone}, {whale.label}: <b className="font-semibold">{(whale.action ?? "no action on record").replace(/\.+$/, "")}</b>.
        </p>
        <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[14.5px]">
          <dt className="text-n-ink-3">Depth limit</dt>
          <dd className="text-n-ink">{whale.depth_limit_fathoms != null ? `${whale.depth_limit_fathoms} fathoms` : "none on record"}</dd>
          <dt className="text-n-ink-3">Gear reduction</dt>
          <dd className="text-n-ink">{whale.gear_reduction_pct != null ? `${whale.gear_reduction_pct}% fewer traps` : "none on record"}</dd>
          {whale.last_assessment && (
            <>
              <dt className="text-n-ink-3">Last assessment</dt>
              <dd className="text-n-ink">{dayWords(whale.last_assessment, now)}</dd>
            </>
          )}
          <dt className="text-n-ink-3">Next assessment</dt>
          <dd className="text-n-ink" data-testid="whale-next">
            {whale.next_assessment ? dayWords(whale.next_assessment, now) : "not announced"}
            {whale.next_window && <span className="block text-[13px] text-n-ink-3">CDFW: {whale.next_window}</span>}
          </dd>
        </dl>
        <p className="today-src mt-2">
          CDFW Risk Assessment Mitigation Program{whale.season ? `, ${whale.season} season` : ""} ·{" "}
          <SourceLink name="source" url={whale.source_url}>
            source
          </SourceLink>{" "}
          · {whale.verified ? "checked by a person against the source" : <NotVerifiedWords />}
        </p>
        <p className="nx-margin mt-2">About whale entanglement in trap gear, not toxin. Neritic has no whale-sighting feed for these waters.</p>
      </div>
    </div>
  );
}

export function WhaleSheet({ open, whale, now, onClose }: { open: boolean; whale: WhaleView | null; now: string; onClose: () => void }) {
  return (
    <Sheet open={open && !!whale} onClose={onClose} title="Whale rules (RAMP)" testid="whale-sheet">
      {whale && <WhaleDetail whale={whale} now={now} />}
    </Sheet>
  );
}

export function StatusSheet({ row, whale, now, onClose }: { row: StatusRow | null; whale: WhaleView | null; now: string; onClose: () => void }) {
  const r = row;
  // kept through the exit, so the sheet does not empty while it leaves
  const [last, setLast] = useState<StatusRow | null>(r);
  if (r && r !== last) setLast(r);
  const shown = r ?? last;
  return (
    <Sheet open={!!r} onClose={onClose} title={shown?.label ?? ""} head={shown ? <RowTag r={shown} testId="sheet-word" /> : undefined} testid="status-sheet">
      {shown && (
        <div>
          <p className="text-[17px] leading-snug text-n-ink">{rowHeadline(shown)}</p>
          {shown.fact.detail && <p className="mt-2 text-[15px] leading-relaxed text-n-ink-2">{shown.conflict ? shown.fact.detail.replace(shown.conflict.words, "").trim() : shown.fact.detail}</p>}
          {shown.conflict && (
            <div className="mt-4" data-testid="conflict">
              <h3 className="nx-caps text-n-stale">The sources differ</h3>
              <p className="mt-1.5 text-[14.5px] leading-relaxed text-n-ink">Neritic has two records for this fishery here and they do not agree. It treats the fishery by the more restrictive one and does not choose between them: a person has to check both.</p>
              <dl className="mt-2 grid gap-3 text-[14.5px] leading-snug">
                {(
                  [
                    ["notice", "Official-notices registry", `${shown.conflict.notice.word}: ${shown.conflict.notice.title}`, [shown.conflict.notice.agency, shown.conflict.notice.effective ? `in effect since ${dayWords(shown.conflict.notice.effective, now)}` : null, shown.conflict.notice.checked ? `registry last read ${dayWords(shown.conflict.notice.checked, now)}` : "read date not recorded", shown.conflict.notice.verified ? "verified" : "not verified"], shown.conflict.notice.url],
                    ["record", "Neritic's curated status", `${shown.conflict.record.word}: ${shown.conflict.record.headline}`, [shown.conflict.record.source, `checked ${dayWords(shown.conflict.record.checked, now)}`, shown.conflict.record.verified ? "verified" : "not verified"], shown.conflict.record.url],
                  ] as const
                ).map(([key, name, says, when, url]) => (
                  <div key={key} data-testid={`conflict-${key}`}>
                    <dt className="nx-caps">
                      {name}
                      {shown.conflict!.shown === key ? " · the more restrictive" : ""}
                    </dt>
                    <dd className="mt-0.5 text-n-ink">
                      {says}
                      <span className="block text-[13px] text-n-ink-3">
                        {when.filter(Boolean).join(" · ")}
                        {url && (
                          <>
                            {" · "}
                            <a href={url} target="_blank" rel="noreferrer" className="nx-link nx-focus">
                              source
                            </a>
                          </>
                        )}
                      </span>
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
          <Dl
            rows={[
              ...(shown.seasonal ? ([["Why", "Closed by its season dates, not by an emergency action. It is not open."]] as [string, ReactNode][]) : shown.causeWord ? ([["Cause", shown.causeWord]] as [string, ReactNode][]) : []),
              ...(shown.area ? ([["Area", shown.area]] as [string, ReactNode][]) : []),
              ...(shown.records[0]?.effective ? ([["Effective", dayWords(shown.records[0].effective, now)]] as [string, ReactNode][]) : []),
              ...(shown.records[0]?.until ? ([["Until", dayWords(shown.records[0].until, now)]] as [string, ReactNode][]) : []),
              ...(shown.next_review ? ([["Next review", dayWords(shown.next_review, now)]] as [string, ReactNode][]) : []),
              ...(shown.source ? ([["Source", <SourceLink key="s" name={shown.source.name} url={shown.source.url} />]] as [string, ReactNode][]) : []),
              ...(shown.checked_at ? ([["Checked", dayWords(shown.checked_at, now)]] as [string, ReactNode][]) : []),
              ["Review", shown.verified === true ? "Checked by a person against the source" : <span key="v" className="text-n-official">{shown.verified === false ? "Not verified: no person has checked this record against the source yet" : "Not verified: Neritic holds no record for this fishery here, so no person has checked anything"}</span>],
            ]}
          />
          {shown.records.length > 1 && (
            <>
              <h3 className="nx-caps mt-6">Also on record</h3>
              <ul className="mt-2">
                {shown.records.slice(1).map((x) => (
                  <li key={x.id} className="border-t border-n-line py-2 text-[14.5px] text-n-ink-2">
                    {x.headline}
                  </li>
                ))}
              </ul>
            </>
          )}
          {shown.notices.length > 0 && (
            <>
              <h3 className="nx-caps mt-6">Official notices for this fishery here</h3>
              <ul className="mt-2">
                {shown.notices.map((n) => (
                  <li key={n.id} className="border-t border-n-line py-2.5 text-[14.5px]">
                    <span className="text-n-ink">
                      {n.agency}: {n.title}
                    </span>{" "}
                    {n.verified ? null : <NotVerifiedWords />}
                    {n.url && (
                      <a href={n.url} target="_blank" rel="noreferrer" className="nx-link nx-focus ml-1 inline-flex min-h-11 items-center gap-1">
                        source <Icon name="external" className="h-3.5 w-3.5" />
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            </>
          )}
          {whale && /^dungeness/.test(shown.fishery) && (
            <>
              <h3 className="nx-caps mt-6">Whale rules (RAMP)</h3>
              <WhaleDetail whale={whale} now={now} />
            </>
          )}
          <p className="nx-margin mt-6">Neritic is not an official source. CDFW and CDPH decide closures and reopenings; call before you go: CDFW closure line (831) 649-2883, CDPH biotoxin line 1 (800) 553-4133.</p>
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
    <Sheet open={open} onClose={onClose} title="Fisheries you follow" testid="fisheries-sheet">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSave(FISHERY_OPTIONS.map((o) => o.id).filter((id) => pick.includes(id)));
        }}
      >
        <fieldset>
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
        <div className="mt-4">
          <Button type="submit" kind="primary" disabled={!pick.length} testId="fisheries-save" className="w-full">
            Show these
          </Button>
        </div>
      </form>
    </Sheet>
  );
}
