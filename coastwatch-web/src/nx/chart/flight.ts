"use client";

import { FLIGHT_NEAR, ticketOut } from "@/nx/ds/motion";

/**
 * The page change's sequence (v14 README §5, SYSTEM.md "Screen change"): the old ticket shrinks toward its anchor
 * (170 ms) → the camera flies → the new ticket grows from its pin at 62 % of the flight → its dock springs up 70 ms
 * after. The chart host writes each flight here; tickets and docks wait on it. Nothing here touches the map.
 */
type Flight = { start: number; duration: number };
let flight: Flight | null = null;
let leavingUntil = 0;
const now = () => (typeof performance === "undefined" ? 0 : performance.now());

/** A ticket is leaving (ds Ticket's ghost): a flight that starts now waits for it to finish shrinking. */
export function noteTicketLeaving() {
  leavingUntil = Math.max(leavingUntil, now() + (ticketOut.duration as number) * 1000);
}
/** How long the camera should wait before it leaves, ms (0 when no ticket is leaving). */
export const flightDelay = () => Math.max(0, leavingUntil - now());

/** The chart host: a flight starts `delay` ms from now and lasts `duration` ms (0 for a jump). */
export function noteFlight(delay: number, duration: number) {
  flight = duration > 0 ? { start: now() + delay, duration } : null;
}

/**
 * Resolves when the flight under way (or about to start) is `at` of the way through (62 % by default), at once when
 * the camera is still. A ticket calls it on arrival, so it grows from its pin as the camera nears the place.
 */
export function whenFlightNear(at = FLIGHT_NEAR): Promise<void> {
  const f = flight;
  if (!f) return Promise.resolve();
  const wait = f.start + f.duration * at - now();
  if (wait <= 4) return Promise.resolve();
  return new Promise((res) => setTimeout(res, wait));
}

/** True while the camera is flying between places (the deep-water texture hides then). */
export const flying = () => !!flight && now() < flight.start + flight.duration;
