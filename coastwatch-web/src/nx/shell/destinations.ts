/**
 * The five sections (v14: Today / My Coast / Research / Alerts; v15.1 adds Explore, the statewide ocean view) and
 * which one a pathname belongs to. DOM- and style-free (unit tests import it). Every official notice is reached
 * from Today, so it keeps Today current.
 */
export const DESTINATIONS = [
  { path: "/today", label: "Today" },
  { path: "/explore", label: "Explore" },
  { path: "/coast", label: "My Coast" },
  { path: "/research", label: "Research" },
  { path: "/alerts", label: "Alerts" },
] as const;

/** The section a pathname belongs to; `/` is Today for a reader with a saved port, else none. */
export function activeDestination(pathname: string, savedPort: string | null | undefined): string | null {
  if (pathname === "/") return savedPort ? "/today" : null;
  if (pathname === "/notices") return "/today";
  return DESTINATIONS.find((d) => pathname === d.path || pathname.startsWith(`${d.path}/`))?.path ?? null;
}

/** A section's address carries the current port, so the port travels across the sections. */
export const withPlace = (path: string, key: string | null) => (key ? `${path}?place=${encodeURIComponent(key)}` : path);

/**
 * A section's link. Explore opens on the whole coast from the navigation (Explore California Waters), with the port
 * kept for the way back; the other sections carry the port only.
 */
export const destinationHref = (path: string, key: string | null) => (path === "/explore" ? `/explore?view=coast${key ? `&place=${encodeURIComponent(key)}` : ""}` : withPlace(path, key));
