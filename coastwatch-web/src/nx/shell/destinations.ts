/**
 * The four sections (v14: Today / My Coast / Research / Alerts) and which one a pathname belongs to. DOM- and
 * style-free (unit tests import it). Explore is Today's own data layers and time (reached from Today and the menu),
 * and every official notice is reached from Today, so both keep Today current.
 */
export const DESTINATIONS = [
  { path: "/today", label: "Today" },
  { path: "/coast", label: "My Coast" },
  { path: "/research", label: "Research" },
  { path: "/alerts", label: "Alerts" },
] as const;

/** The section a pathname belongs to; `/` is Today for a reader with a saved port, else none. */
export function activeDestination(pathname: string, savedPort: string | null | undefined): string | null {
  if (pathname === "/") return savedPort ? "/today" : null;
  if (pathname === "/notices" || pathname === "/explore" || pathname.startsWith("/explore/")) return "/today";
  return DESTINATIONS.find((d) => pathname === d.path || pathname.startsWith(`${d.path}/`))?.path ?? null;
}

/** A section's address carries the current port, so the port travels across the four. */
export const withPlace = (path: string, key: string | null) => (key ? `${path}?place=${encodeURIComponent(key)}` : path);
