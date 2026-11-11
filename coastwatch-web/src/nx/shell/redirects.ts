/**
 * Old addresses, sent to the one canonical home of each view (journeys.md Architecture; research C §6
 * "Redirect plan"). Pure, unit tested; the pages call them on the server and answer with a permanent redirect.
 */
type SP = Record<string, string | string[] | undefined>;

/** Parameters that say nothing about what to open (a shared link's campaign tags). */
const CAMPAIGN = /^(utm_[a-z_]+|fbclid|gclid|msclkid|mc_[a-z]+|ref|source)$/i;
/** The front door's own parameters (`pick=1` opens the port finder). */
const FRONT = /^(pick)$/;

const entries = (sp: SP) => Object.entries(sp).flatMap(([k, v]) => (v === undefined ? [] : Array.isArray(v) ? v.map((x) => [k, x] as const) : [[k, v] as const]));

const qs = (pairs: ReadonlyArray<readonly [string, string]>) => {
  const u = new URLSearchParams();
  for (const [k, v] of pairs) u.append(k, v);
  const s = u.toString();
  return s ? `?${s}` : "";
};

/**
 * `/` with anything that names a view (Explore's own parameters, `?view=map`) was Explore's address: it is
 * now `/explore`, with every parameter kept but `view`. A bare `/`, or one carrying only campaign tags, is the
 * front door and stays.
 */
export function rootRedirect(sp: SP): string | null {
  const pairs = entries(sp);
  if (!pairs.some(([k]) => !CAMPAIGN.test(k) && !FRONT.test(k))) return null;
  return `/explore${qs(pairs.filter(([k]) => k !== "view"))}`;
}

/** `/bloom?station&var&range&year` was the v2 station page: Research, with `var` as `variable` and a year as its window. */
export function bloomRedirect(sp: SP): string {
  const out: Array<[string, string]> = [];
  for (const [k, v] of entries(sp)) {
    if (k === "var") out.push(["variable", v]);
    else if (k === "year" && /^\d{4}$/.test(v)) out.push(["start", `${v}-01-01`], ["end", `${v}-12-31`]);
    else if (k !== "year") out.push([k, v]);
  }
  return `/research${qs(out)}`;
}

/** `/coast?mode=research&place=X` asked for the Research view of a place. */
export function coastRedirect(sp: SP): string | null {
  if (sp.mode !== "research") return null;
  return `/research${typeof sp.place === "string" ? qs([["place", sp.place]]) : ""}`;
}

const DESTINATIONS = new Set(["/today", "/explore", "/coast", "/research"]);

/** Where choosing a port in the bar's finder takes the reader: the same page at the new port; from `/`, Today. */
export function hrefForPort(pathname: string, key: string): string {
  const path = DESTINATIONS.has(pathname) ? pathname : "/today";
  return `${path}?place=${encodeURIComponent(key)}`;
}
