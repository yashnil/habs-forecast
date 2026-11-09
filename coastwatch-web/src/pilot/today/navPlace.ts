/** The last port opened in Today, on this device only (kept tiny: the top bar of every page reads it). */
export const TODAY_PLACE_KEY = "nx-today-place-v1";
const PLACE = /^port-\d{1,4}$/;

export function readTodayPlace(): string | null {
  try {
    const v = window.localStorage.getItem(TODAY_PLACE_KEY);
    return v && PLACE.test(v) ? v : null;
  } catch {
    return null;
  }
}
export function writeTodayPlace(place: string): boolean {
  if (!PLACE.test(place)) return false;
  try {
    window.localStorage.setItem(TODAY_PLACE_KEY, place);
    return true;
  } catch {
    return false;
  }
}
/** The nav's link: the last port opened in Today, or null (plain /today). */
export const readTodayHref = (): string | null => {
  const p = readTodayPlace();
  return p ? `/today?place=${p}` : null;
};
