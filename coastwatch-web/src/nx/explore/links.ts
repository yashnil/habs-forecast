/**
 * The address of Explore California Waters (v15.1), DOM-free so every page that links here builds the same URL
 * and the unit tests read it. The contract other pages may rely on:
 *
 *   /explore?view=coast&place=<port-N>&layer=<chl|charm>&cur=<0|1>&st=1&at=<lat>,<lon>&from=<today|trip|research>&trip=<id>
 *
 * - `view=coast`: open on the whole coast, Oregon border to Mexico (the camera flies there from wherever the chart
 *   was). Without it Explore opens on the port's water, as it always has.
 * - `place`: the reader's port; kept, so "Back to <port>" returns to its water and every link carries it on.
 * - `layer`: the surface layer (chlorophyll, the default, or C-HARM). `cur=1` currents on, `st=1` stations on.
 * - `at`: a point to read on arrival (a cell of the surface layer).
 * - `from` / `trip`: where "Back to …" goes: Today (the default), Research, or one saved trip (`/coast/trips?id=`).
 *
 * Unknown values are ignored, never guessed at.
 */
export type ExploreView = "coast" | "port";
export type ExploreFrom = { kind: "today" } | { kind: "research" } | { kind: "trip"; id: string };

export type ExploreLink = {
  place?: string | null;
  view?: ExploreView;
  layer?: "chl" | "charm";
  currents?: boolean;
  stations?: boolean;
  at?: { lat: number; lon: number } | null;
  from?: "today" | "research" | "trip";
  trip?: string | null;
};

export function exploreHref(o: ExploreLink = {}): string {
  const q = new URLSearchParams();
  if (o.view === "coast") q.set("view", "coast");
  if (o.place) q.set("place", o.place);
  if (o.layer && o.layer !== "chl") q.set("layer", o.layer);
  if (o.currents !== undefined) q.set("cur", o.currents ? "1" : "0");
  if (o.stations) q.set("st", "1");
  if (o.at) q.set("at", `${o.at.lat.toFixed(4)},${o.at.lon.toFixed(4)}`);
  if (o.from === "trip" && o.trip) {
    q.set("from", "trip");
    q.set("trip", o.trip);
  } else if (o.from === "research") q.set("from", "research");
  const s = q.toString();
  return s ? `/explore?${s}` : "/explore";
}

/** Where the reader came from, from the address (`from`, `trip`). A trip without its id is Today. */
export function readFrom(sp: URLSearchParams): ExploreFrom {
  const f = sp.get("from");
  const trip = sp.get("trip");
  if (f === "trip" && trip && /^[\w-]{1,80}$/.test(trip)) return { kind: "trip", id: trip };
  if (f === "research") return { kind: "research" };
  return { kind: "today" };
}

/** The view the address asks for. */
export const readView = (sp: URLSearchParams): ExploreView => (sp.get("view") === "coast" ? "coast" : "port");

/** "Back to …": the page the reader came from, carrying the port. */
export function backLink(from: ExploreFrom, place: string | null): { href: string; label: string } {
  const withPlace = (p: string) => (place ? `${p}?place=${encodeURIComponent(place)}` : p);
  if (from.kind === "trip") return { href: `/coast/trips?id=${encodeURIComponent(from.id)}`, label: "Back to your trip" };
  if (from.kind === "research") return { href: withPlace("/research"), label: "Back to Research" };
  return { href: withPlace("/today"), label: "Back to Today" };
}
