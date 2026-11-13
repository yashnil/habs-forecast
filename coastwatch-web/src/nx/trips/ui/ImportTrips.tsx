"use client";

import { useRef, useState } from "react";
import { Button, Dialog } from "@/nx/ds";
import type { ImportPreview, TripsStore } from "../store";
import { notifyTrips, tripsStore } from "../useTripsData";
import s from "./economics.module.css";

export const IMPORT_WORDS: Record<string, string> = {
  invalid: "That file is not a Neritic trips export. Nothing was imported.",
  newer: "That file was written by a newer version of Neritic. Nothing was imported.",
  quota: "This browser's storage is full. Nothing more was imported.",
  readOnly: "A newer version of Neritic manages the trips on this device. Nothing was imported.",
  unavailable: "This browser is not keeping data for Neritic (private window or storage blocked).",
  consent: "Nothing was imported.",
  missing: "Nothing was imported.",
};

const n = (k: number, one: string, many: string) => `${k} ${k === 1 ? one : many}`;

/** The consent's lines: what this file would really do, from the store's preview. */
export function previewLines(p: Extract<ImportPreview, { ok: true }>): string[] {
  const out: string[] = [];
  if (p.fresh) out.push(`${n(p.fresh, "new trip", "new trips")} to add.`);
  if (p.copies) out.push(`${n(p.copies, "trip shares", "trips share")} an id with a different trip here: ${p.copies === 1 ? "it comes" : "they come"} in as a separate copy named "(imported)".`);
  if (p.already) out.push(`${n(p.already, "trip is", "trips are")} already on this device and will be skipped.`);
  if (p.invalid) out.push(`${n(p.invalid, "record", "records")} cannot be read and will be left out.`);
  return out;
}

export type ImportTripsProps = {
  /** defaults to this tab's store */
  store?: TripsStore;
  /** after a finished import (defaults to notifying every view of this tab) */
  onDone?: (added: number) => void;
  /** the trigger's words */
  label?: string;
  variant?: "quiet" | "secondary";
};

/**
 * Import a trips export (.json): pick a file, it is read and checked on this device first; a file that is not an
 * export, or holds nothing new, says so and never asks for consent. Otherwise one dialog states exactly what would
 * happen (new trips, trips already here, copies, unreadable records) and imports only on the reader's Import. A report
 * follows. Nothing is uploaded. Self-contained, so the empty state can offer it too (restoring a backup on a new phone).
 */
export function ImportTrips({ store: given, onDone, label = "Import", variant = "quiet" }: ImportTripsProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [pending, setPending] = useState<{ name: string; text: string; lines: string[] } | null>(null);
  const [report, setReport] = useState<{ title: string; lines: string[] } | null>(null);
  const store = () => given ?? tripsStore();

  const pick = async (file: File | undefined) => {
    if (fileRef.current) fileRef.current.value = "";
    if (!file) return;
    let text = "";
    try {
      text = await file.text();
    } catch {
      setReport({ title: "Nothing was imported", lines: ["That file could not be read."] });
      return;
    }
    const p = store().previewImport(text);
    if (!p.ok) {
      setReport({ title: "Nothing was imported", lines: [IMPORT_WORDS[p.reason] ?? "That file could not be read."] });
      return;
    }
    if (!p.fresh && !p.copies) {
      setReport({ title: "Nothing new in this file", lines: [...previewLines(p), ...p.errors] });
      return;
    }
    setPending({ name: file.name, text, lines: previewLines(p) });
  };

  const confirm = () => {
    if (!pending) return;
    const r = store().importAll(pending.text, { consent: true });
    setPending(null);
    if (!r.ok) setReport({ title: "Nothing was imported", lines: [IMPORT_WORDS[r.reason] ?? "That file could not be read."] });
    else {
      setReport({
        title: r.added === 1 ? "1 trip imported" : `${r.added} trips imported`,
        lines: [...(r.reassigned ? [`${n(r.reassigned, "trip", "trips")} came in as a separate copy named "(imported)".`] : []), ...(r.already ? [`${n(r.already, "trip was", "trips were")} already on this device and skipped.`] : []), ...r.errors],
      });
      if (onDone) onDone(r.added);
      else notifyTrips();
    }
  };

  return (
    <>
      <Button variant={variant} onClick={() => fileRef.current?.click()} data-testid="trips-import">
        {label}
      </Button>
      <input ref={fileRef} type="file" accept="application/json,.json" className={s.vh} tabIndex={-1} aria-hidden="true" data-testid="trips-import-file" onChange={(e) => void pick(e.target.files?.[0])} />
      <Dialog
        open={!!pending}
        onOpenChange={(o) => (o ? null : setPending(null))}
        title="Import trips from this file?"
        description={pending?.name}
        footer={
          <div className={s.sheetFoot}>
            <Button variant="quiet" onClick={() => setPending(null)}>
              Cancel
            </Button>
            <Button variant="primary" glyph="↓" onClick={confirm} data-testid="trips-import-confirm">
              Import
            </Button>
          </div>
        }
      >
        <ul className={s.consent} data-testid="trips-import-counts">
          {pending?.lines.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
        <p className={s.note}>The trips are added to this device only. Nothing is uploaded, and nothing already here is replaced.</p>
      </Dialog>
      <Dialog open={!!report} onOpenChange={(o) => (o ? null : setReport(null))} title={report?.title ?? ""} footer={<Button onClick={() => setReport(null)} data-testid="trips-import-close">Close</Button>}>
        {report?.lines.length ? (
          <ul className={s.consent} data-testid="trips-import-report">
            {report.lines.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        ) : null}
      </Dialog>
    </>
  );
}
