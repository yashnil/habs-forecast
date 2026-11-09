import { PILOT_PORTS, type PortToday } from "@/pilot/contracts";
import { stubPortToday } from "./factsStub";

/**
 * Where alerts get the current PortToday facts (server side: the /account page and alerts-fanout).
 *
 * INTEGRATOR: when src/pilot/facts.ts (pilot/today) is merged, replace the body with the real
 * builder and return `stub: false`, e.g.
 *   const todays = await Promise.all(places.map((p) => buildPortToday(p)));
 *   return { todays: todays.filter(Boolean), stub: false };
 * Nothing else in accounts/alerts changes: they only read PortToday (contracts.ts).
 */
export type TodaysResult = { todays: PortToday[]; stub: boolean };

export async function loadPortTodays(places: readonly string[] = PILOT_PORTS.map((p) => p.place_id)): Promise<TodaysResult> {
  const todays = places.map((p) => stubPortToday(p)).filter((t): t is PortToday => t !== null);
  return { todays, stub: true };
}
