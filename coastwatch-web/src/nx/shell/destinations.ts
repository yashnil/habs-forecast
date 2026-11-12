/** The four destinations and which one a pathname belongs to. DOM- and style-free (unit tests import it). */
export const DESTINATIONS = [
  { path: "/today", label: "Today" },
  { path: "/explore", label: "Explore" },
  { path: "/coast", label: "My Coast" },
  { path: "/research", label: "Research" },
] as const;

/** The page a pathname belongs to; `/` is Today for a reader with a saved port, else no tab. */
export function activeDestination(pathname: string, savedPort: string | null | undefined): string | null {
  if (pathname === "/") return savedPort ? "/today" : null;
  // every official notice is reached from Today (its Notices row): Today stays the current page
  if (pathname === "/notices") return "/today";
  return DESTINATIONS.find((d) => pathname === d.path || pathname.startsWith(`${d.path}/`))?.path ?? null;
}

/** A tab's address carries the current port, so the port travels across the four pages. */
export const withPlace = (path: string, key: string | null) => (key ? `${path}?place=${encodeURIComponent(key)}` : path);
