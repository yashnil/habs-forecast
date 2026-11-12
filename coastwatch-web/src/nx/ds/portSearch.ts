/**
 * Ranking for the PortFinder (spec section 9): what a person types is a port name first. A name that starts
 * with the query beats a name with a word that starts with it, which beats a name that merely contains it.
 * County and the CDFW port code are keywords, ranked after every name match. Region names are never matched,
 * and letters are never matched out of order, so "moss" finds Moss Landing and "zzz" finds nothing.
 *
 * Kept free of React and styles so the unit suite can import it.
 */

export interface SearchablePort {
  code: number;
  name: string;
  county: string;
  lat: number;
}

/** Lower case, no accents, one space between words, no punctuation. */
export function normalizeQuery(text: string): string {
  return text
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

/** Match tiers, best first; `null` is no match. */
export const TIER = {
  namePrefix: 0,
  nameWordPrefix: 1,
  nameSubstring: 2,
  countyWordPrefix: 3,
  code: 4,
} as const;

function wordPrefix(hay: string, q: string): boolean {
  return hay.split(" ").some((w, i, words) => words.slice(i).join(" ").startsWith(q));
}

/** Every word of the query starts a word of the haystack ("moss la" → "moss landing"). */
function allWordsPrefix(hay: string, q: string): boolean {
  const words = hay.split(" ");
  return q.split(" ").every((t) => words.some((w) => w.startsWith(t)));
}

export function portTier(port: SearchablePort, query: string): number | null {
  const q = normalizeQuery(query);
  if (!q) return null;
  const name = normalizeQuery(port.name);
  const county = normalizeQuery(port.county);
  if (name.startsWith(q)) return TIER.namePrefix;
  if (wordPrefix(name, q) || allWordsPrefix(name, q)) return TIER.nameWordPrefix;
  if (q.length >= 2 && name.includes(q)) return TIER.nameSubstring;
  if (wordPrefix(county, q) || allWordsPrefix(`${name} ${county}`, q)) return TIER.countyWordPrefix;
  if (/^\d+$/.test(q) && String(port.code) === q) return TIER.code;
  return null;
}

/**
 * The ports that match `query`, best first: by tier, then by where the match falls in the name, then north
 * to south. An empty query returns an empty list (the finder shows its directory instead).
 */
export function rankPorts<P extends SearchablePort>(ports: readonly P[], query: string): P[] {
  const q = normalizeQuery(query);
  if (!q) return [];
  return ports
    .map((p) => ({ p, tier: portTier(p, q), at: normalizeQuery(p.name).indexOf(q) }))
    .filter((r): r is { p: P; tier: number; at: number } => r.tier !== null)
    .sort((a, b) => a.tier - b.tier || (a.at < 0 ? 1e3 : a.at) - (b.at < 0 ? 1e3 : b.at) || b.p.lat - a.p.lat || a.p.name.localeCompare(b.p.name))
    .map((r) => r.p);
}
