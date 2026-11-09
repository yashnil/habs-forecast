import { PILOT_PORTS, type PortToday } from "@/pilot/contracts";
import { loadPortTodayFacts } from "@/pilot/today/portToday";

/**
 * Where alerts get the current PortToday facts (server side: the /account page and alerts-fanout):
 * the same facts Today shows (src/pilot/today/portToday.ts), every fishery included. The sea is not
 * fetched: no alert trigger reads sea, tide or buoy facts.
 *
 * `sample` is true when the server is reading the test fixture (never real conditions): pages say so,
 * and the fan-out never sends such facts to a phone. A port whose data cannot be loaded is simply
 * absent (the fan-out reports it as a place without facts; nothing is invented).
 */
export type TodaysResult = { todays: PortToday[]; sample: boolean };

export async function loadPortTodays(places: readonly string[] = PILOT_PORTS.map((p) => p.place_id)): Promise<TodaysResult> {
  const loaded = (await Promise.all(places.map((p) => loadPortTodayFacts(p, { sea: false })))).filter((x) => x !== null);
  return { todays: loaded.map((x) => x.today), sample: loaded.some((x) => x.fixture) };
}
