"use client";

import { Button, Micro } from "@/nx/ds";
import { ImportTrips } from "./ImportTrips";
import { TripsUndo } from "./TripsUndo";
import s from "./trips.module.css";

/**
 * No trips on this device yet: the chart behind, a small chartroom figure (a sounding line over isobaths, the
 * reader's trip still to be drawn), one line and one action; quietly, restoring a backup, and Undo right after the last
 * trip was deleted.
 */
export function EmptyTrips() {
  return (
    <div className={s.empty} data-testid="trips-empty">
      {/* the last trip just deleted: put back from here (C keeps the pending undo outside the list) */}
      <TripsUndo />
      <svg className={s.emptyArt} viewBox="0 0 240 96" aria-hidden="true">
        <path d="M4 70 C 50 58, 90 80, 140 66 S 220 52, 236 60" className={s.iso} />
        <path d="M4 50 C 46 40, 96 60, 146 46 S 214 34, 236 40" className={s.iso} />
        <path d="M4 30 C 52 22, 100 38, 150 26 S 210 16, 236 20" className={s.isoFaint} />
        <path d="M40 84 L 120 48 L 196 30" className={s.course} />
        <circle cx="40" cy="84" r="5" className={s.portDot} />
        <circle cx="196" cy="30" r="4" className={s.markHollow} />
      </svg>
      <Micro>Your trips</Micro>
      <h1 id="trips-title" className={s.emptyTitle}>
        Plan a trip, then come back to it.
      </h1>
      <Button variant="primary" href="/coast?view=trip" data-testid="trips-plan">
        Plan a trip
      </Button>
      <ImportTrips label="Restore from a backup" variant="quiet" />
    </div>
  );
}
