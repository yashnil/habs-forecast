/**
 * Whale measures (CDFW RAMP) for one reader, the ONE formatter Today and My Coast share, so the two pages
 * cannot disagree (T41, T42). PURE.
 *
 * The data gives each zone two structured fields (`depth_limit_fathoms`, `gear_reduction_pct`, both RAMP
 * tools for the commercial trap fleet) and one free-text `action` from CDFW ("Commercial season delayed to
 * at least Jan 1, 2027; recreational crab traps prohibited; Fleet Advisory."). Neritic does not paraphrase
 * that text: it splits it at its semicolons and gives each reader only the clauses that name their sector,
 * in CDFW's own words. A clause that names neither sector is shown to both, except a Fleet Advisory, which
 * RAMP addresses to the commercial fleet. A commercial crabber is never told the recreational rule, and a
 * recreational crabber never gets the commercial fleet's depth or gear figures.
 */

export type Sector = "commercial" | "recreational";

export type WhaleZoneFields = {
  zone: string;
  action: string | null;
  depth_limit_fathoms: number | null;
  gear_reduction_pct: number | null;
};

export type WhaleMeasures = {
  /** one short line for a row's value: "Fleet Advisory, no depth or gear limit on record" */
  short: string;
  /** full sentences for this reader: CDFW's clauses for the sector, then the structured fields */
  sentences: string[];
  /** CDFW's clauses for this sector, verbatim, for a quotation with its source; null when none */
  quote: string | null;
  /** the structured fields (or their absence) in words, for showing after the quotation */
  fields: string[];
};

const clean = (c: string) => c.trim().replace(/\s+/g, " ").replace(/[.]+$/, "");
const cap1 = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const lower1 = (s: string) => (s && !/^[A-Z]{2}|^Fleet Advisory/.test(s) ? s.charAt(0).toLowerCase() + s.slice(1) : s);
const sentence = (s: string) => `${cap1(s)}.`;

/** Which reader a clause of CDFW's zone action is addressed to. */
export function clauseSector(clause: string): Sector | "both" {
  const rec = /\brecreational\b/i.test(clause);
  const com = /\bcommercial\b/i.test(clause) || /\bfleet advisory\b/i.test(clause);
  if (rec && !com) return "recreational";
  if (com && !rec) return "commercial";
  return "both";
}

/** The clauses of CDFW's zone action, split at semicolons and sentence ends, without trailing stops. */
export function actionClauses(action: string | null | undefined): string[] {
  return (action ?? "")
    .split(/;\s*|\.\s+(?=[A-Z])/)
    .map(clean)
    .filter(Boolean);
}

/**
 * The measures in a zone for one reader.
 * `seasonSaysIt`: the season line on the same page already states a whale delay or closure, so that clause
 * is not repeated (one home per fact, README principle 5).
 */
export function whaleMeasures(z: WhaleZoneFields, sector: Sector, opts: { seasonSaysIt?: boolean } = {}): WhaleMeasures {
  const mine = actionClauses(z.action).filter((c) => {
    const s = clauseSector(c);
    return s === sector || s === "both";
  });
  const quote = mine.length ? `${cap1(mine.join("; "))}.` : null;
  const shown = mine.filter((c) => !(opts.seasonSaysIt && /\bseason\b.*\b(delayed|closed)\b/i.test(c)));
  const sentences = shown.map(sentence);
  const fields: string[] = [];
  const shortParts = shown.map(lower1);

  if (sector === "commercial") {
    const depth = z.depth_limit_fathoms != null ? `traps ${z.depth_limit_fathoms} fathoms or shallower` : null;
    const gear = z.gear_reduction_pct != null ? `trap numbers cut by ${z.gear_reduction_pct} %` : null;
    for (const x of [depth, gear]) if (x) fields.push(sentence(x));
    if (depth) shortParts.push(`${z.depth_limit_fathoms} fathom limit`);
    if (gear) shortParts.push(`traps cut ${z.gear_reduction_pct} %`);
    if (!depth && !gear) {
      fields.push("No depth limit or gear reduction on record.");
      shortParts.push("no depth or gear limit on record");
    }
  } else if (!shown.length) {
    fields.push("CDFW's zone action names no recreational measure.");
    shortParts.push("no recreational measure on record");
  }
  return { short: cap1(shortParts.join(", ")), sentences: [...sentences, ...fields], quote, fields };
}

/** The reader a Dungeness fishery id stands for. */
export const sectorOf = (fishery: string): Sector => (/recreational/.test(fishery) ? "recreational" : "commercial");
