import { isPortKey, readPortRecord, updatePortRecord } from "@/lib/portRecord";

/** The last port opened in Today, on this device only, in the saved-port record (kept tiny: the top bar of every page reads it). */
export function readTodayPlace(): string | null {
  return readPortRecord().lastPort;
}
export function writeTodayPlace(place: string): boolean {
  if (!isPortKey(place)) return false;
  return updatePortRecord((r) => ({ ...r, lastPort: place }));
}
/** The nav's link: the last port opened in Today, or null (plain /today). */
export const readTodayHref = (): string | null => {
  const p = readTodayPlace();
  return p ? `/today?place=${p}` : null;
};
