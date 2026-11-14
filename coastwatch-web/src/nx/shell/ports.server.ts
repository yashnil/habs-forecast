import { cache } from "react";
import { loadToday } from "@/pilot/today/load";
import type { ShellPort } from "./port";

/**
 * Every California port the shell can open, pilot ports first, then north to south (pilot/today/load.ts
 * `portOptions`). Read once per request; an unreadable dataset gives an empty list and the port control says so.
 */
export const shellPorts = cache(async (): Promise<ShellPort[]> => {
  try {
    const l = await loadToday(null);
    if (!l.ok) return [];
    return l.ports.map((p) => ({ key: p.place_id, name: p.label, lon: p.lon, lat: p.lat, pilot: p.pilot, county: p.county }));
  } catch {
    return [];
  }
});
