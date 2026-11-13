"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { Micro } from "@/nx/ds";
import { swap } from "@/nx/ds/motion";
import { audienceOf, tripFishery } from "@/nx/coast/fisheries";
import { daysUntil, shortDate, STATE_WORDS } from "../lifecycle";
import type { Trip } from "../types";
import s from "./trips.module.css";

/** "tomorrow", "in 3 days", "today", "2 days ago": a departure's distance in the reader's words. */
export function whenWords(t: Trip, now: Date | null): string | null {
  if (!t.depart || !now) return null;
  const d = daysUntil(t, now);
  if (d === null) return null;
  if (d === 0) return "today";
  if (d === 1) return "tomorrow";
  if (d === -1) return "yesterday";
  return d > 0 ? `in ${d} days` : `${-d} days ago`;
}

/**
 * The trip's heading, two short lines: a stamp (state · date, and the port and fishery only when the name does not
 * already say them) and the name, which is itself the rename control (click, type, Enter; Escape keeps the old name).
 * No badges, no verdict.
 */
export function TripHeading({ trip, port, now, onRename, id }: { trip: Trip; port: string | null; now: Date | null; onRename: (name: string) => void; id: string }) {
  const reduce = useReducedMotion();
  const f = tripFishery(trip.fishery);
  const rec = audienceOf(f) === "recreational";
  const rel = whenWords(trip, now);
  const when = trip.depart ? `${shortDate(trip.depart.date)}${trip.depart.time ? ` ${trip.depart.time}` : ""}${rel && trip.state === "planned" ? ` (${rel})` : ""}` : "No date";
  const lower = trip.name.toLowerCase();
  const extra = [port && !lower.includes(port.toLowerCase()) ? port : null, f && !lower.includes(f.label.toLowerCase()) ? f.label : null, rec ? "recreational" : null].filter(Boolean);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(trip.name);
  const input = useRef<HTMLInputElement>(null);
  const title = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);
  const closing = useRef(false);
  const done = (keep: boolean) => {
    // Escape then the blur of the unmounting input: only the first decision counts
    if (closing.current) return;
    closing.current = true;
    setEditing(false);
    if (keep && draft.trim() !== trip.name) onRename(draft);
    requestAnimationFrame(() => title.current?.focus());
  };
  return (
    <header className={s.heading} data-nx-desk-header="" data-testid="trip-heading">
      <Micro className={s.stamp} data-state={trip.state}>
        <AnimatePresence mode="wait" initial={false}>
          <motion.span key={trip.state} className={s.stampState} initial={reduce ? false : { opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={reduce ? undefined : { opacity: 0, y: -4 }} transition={swap.in} data-testid="trip-lifecycle">
            {STATE_WORDS[trip.state]}
          </motion.span>
        </AnimatePresence>
        <span data-testid="trip-when">&nbsp;·&nbsp;{when}</span>
        {extra.length ? <span>&nbsp;·&nbsp;{extra.join(" · ")}</span> : null}
      </Micro>
      <h1 id={id} className={s.name} data-testid="trip-title">
        {editing ? (
          <input
            ref={input}
            className={s.nameInput}
            value={draft}
            maxLength={90}
            aria-label="Trip name"
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => done(true)}
            onKeyDown={(e) => {
              if (e.key === "Enter") done(true);
              if (e.key === "Escape") done(false);
            }}
            data-testid="trip-name-input"
          />
        ) : (
          // design-lint-allow raw-button: the title is its own rename control and must read as the heading, not a button
          <button
            ref={title}
            type="button"
            className={s.nameButton}
            title={`${trip.name}. Click to rename.`}
            aria-label={`${trip.name}. Rename`}
            onClick={() => {
              setDraft(trip.name);
              closing.current = false;
              setEditing(true);
            }}
          >
            {trip.name}
          </button>
        )}
      </h1>
    </header>
  );
}
