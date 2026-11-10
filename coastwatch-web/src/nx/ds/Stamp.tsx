"use client";
import { useState, type ReactNode } from "react";
import { Button } from "./Button";
import { day, dayTime } from "./format";
import { kindLabel, markFor, NOT_VERIFIED, type Kind } from "./kinds";
import { Mark } from "./Mark";
import { Dialog } from "./Overlay";
import { cx } from "./util";
import s from "./Stamp.module.css";

/** Everything Neritic knows about where a number came from. Every number carries one (principle 5). */
export interface Provenance {
  kind: Kind;
  /** What the number is: "Wave height, buoy 46236". Used as the sheet's title. */
  title: string;
  /** The figure as printed, with unit: "6.6 ft at 14 s". */
  value?: string;
  /** Short source for the stamp line: "NDBC buoy 46236". */
  source: string;
  /** Longer source for the sheet: "NDBC 46236 Monterey Canyon Outer, a wave buoy without wind". */
  sourceLong?: string;
  sourceUrl?: string;
  /** When it was measured, issued, sampled, checked or valid (ISO; a time is shown when present). */
  date: string;
  dateRole?: "measured" | "issued" | "sampled" | "checked" | "valid" | "predicted" | "set";
  /** When Neritic read it. */
  retrieved?: string;
  place?: string;
  resolution?: string;
  method?: string;
  caveats?: string[];
  /** Past its stale limit (computed by the caller from the layer's freshness, T17). */
  stale?: boolean;
}

const hasTime = (iso: string) => /T\d{2}:\d{2}/.test(iso) || / \d{2}:\d{2}/.test(iso);
const when = (iso: string, now?: string) => (hasTime(iso) ? dayTime(iso, { ref: now }) : day(iso, { ref: now }));

/** "Measured · NDBC buoy 46236 · Nov 10, 7:00 AM PT" */
export function stampText(p: Provenance, now?: string): string {
  const parts = [kindLabel(p.kind), p.source, `${p.dateRole && p.dateRole !== "measured" ? `${p.dateRole} ` : ""}${when(p.date, now)}`];
  if (p.kind === "official") parts.push(NOT_VERIFIED);
  return parts.join(" · ");
}

export interface StampProps {
  provenance: Provenance;
  now?: string;
  /** When false the stamp is plain text (inside a sheet that already shows provenance). */
  interactive?: boolean;
  className?: string;
}

/**
 * Kind · source · date · verification, the provenance line. It opens the ProvenanceSheet.
 * The kind mark leads, so the class reads before the words.
 */
export function Stamp({ provenance: p, now, interactive = true, className }: StampProps) {
  const [open, setOpen] = useState(false);
  const text = stampText(p, now);
  const inner = (
    <>
      <Mark kind={markFor(p.kind)} withWords={false} />
      <span className={s.text}>{text}</span>
      {p.stale ? (
        <>
          {" "}
          <Mark kind="stale" />
        </>
      ) : null}
    </>
  );
  if (!interactive) return <span className={cx(s.stamp, className)}>{inner}</span>;
  return (
    <>
      <button type="button" className={cx(s.stamp, s.button, className)} onClick={() => setOpen(true)} aria-haspopup="dialog" aria-label={`${text}${p.stale ? ", old" : ""}. Open where this comes from`}>
        {inner}
      </button>
      <ProvenanceSheet provenance={p} now={now} open={open} onOpenChange={setOpen} />
    </>
  );
}

export interface ProvenanceSheetProps {
  provenance: Provenance;
  now?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  trigger?: React.ReactElement;
}

/** The full record behind one number: kind, source, dates, place, method, verification and caveats. */
export function ProvenanceSheet({ provenance: p, now, open, onOpenChange, trigger }: ProvenanceSheetProps) {
  const rows: Array<[string, ReactNode]> = [
    ["Kind", <KindCell key="k" kind={p.kind} />],
    ["Source", p.sourceLong ?? p.source],
    [p.dateRole ? cap(p.dateRole) : "Measured", <span key="d">{when(p.date, now)}{p.stale ? <> <Mark kind="stale" /></> : null}</span>],
  ];
  if (p.retrieved) rows.push(["Read by Neritic", when(p.retrieved, now)]);
  if (p.place) rows.push(["Place", p.place]);
  if (p.resolution) rows.push(["Resolution", p.resolution]);
  if (p.method) rows.push(["Method", p.method]);
  if (p.kind === "official") rows.push(["Verification", "Not verified. Transcribed from the agency's page by machine; no person has checked it yet."]);
  return (
    <Dialog
      placement="side"
      title={p.title}
      open={open}
      onOpenChange={onOpenChange}
      trigger={trigger}
      footer={
        p.sourceUrl ? (
          <Button variant="secondary" href={p.sourceUrl} external icon="share">
            Open the source
          </Button>
        ) : null
      }
    >
      {p.value ? (
        <p className={s.value}>
          {p.value}
        </p>
      ) : null}
      <p className={s.line}>
        <Stamp provenance={p} now={now} interactive={false} />
      </p>
      <dl className={s.rows}>
        {rows.map(([k, v]) => (
          <div key={k} className={s.row}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      {p.caveats && p.caveats.length ? (
        <div className={s.caveats}>
          <h3 className={s.caveatsHead}>Read it with</h3>
          <ul>
            {p.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </Dialog>
  );
}

function KindCell({ kind }: { kind: Kind }) {
  return (
    <span className={s.kindCell}>
      <Mark kind={markFor(kind)} withWords={false} />
      <span>{kind === "official" ? `Official, ${NOT_VERIFIED}` : cap(kind)}</span>
    </span>
  );
}

const cap = (w: string) => w.charAt(0).toUpperCase() + w.slice(1);
