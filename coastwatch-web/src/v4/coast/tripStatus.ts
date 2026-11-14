import type { Audience } from "@/lib/coast";
import type { Species } from "@/content/species";
import type { FisheryId } from "@/pilot/contracts";
import { statusTally, type StatusRow } from "@/pilot/facts";
import type { TagLevel } from "@/v4/ui/Tag";

/**
 * The trip's status line, from the same model as Today (`buildTodayModel` in src/pilot/facts.ts). My Coast
 * never composes a status of its own: it picks Today's row for the fishery the trip names, and words it the
 * same way. Pure, so a unit test can hold the two pages to one answer.
 */

/** Today's fishery for this trip's species and activity, or null when Today has no such fishery row. */
export function fisheryForTrip(species: Species, audience: Audience): FisheryId | null {
  const recreational = audience === "recreational" || audience === "shellfish";
  switch (species.id) {
    case "dungeness":
      return recreational ? "dungeness_recreational" : "dungeness_commercial";
    case "rock-crab":
      return "rock_crab";
    case "salmon":
      // Today follows the commercial salmon season only; a recreational reader has the sport list below
      return recreational ? null : "salmon_commercial";
    case "groundfish":
    case "halibut":
      return "groundfish";
    case "anchovy-sardine":
      return "anchovy_sardine";
    case "squid":
      return "market_squid";
    case "razor-clam":
    case "mussel":
    case "bivalve":
      return "bivalve_shellfish";
    case "lobster":
      return "spiny_lobster";
    default:
      return null;
  }
}

/** The status word's kit tag level; the calmest state is "none on record", never "open". */
export function tagLevel(r: Pick<StatusRow, "level" | "severity" | "seasonal">): TagLevel {
  if (r.level === "closed") return "closed";
  if (r.level === "delayed") return "delayed";
  if (r.level === "no_restriction_on_record" || r.level === "unknown") return "none";
  return r.severity === "amber" || r.severity === "red" ? "restricted" : "watch";
}

export const UNCLEAR_WORDS = "Status unclear, awaiting a person's review";

export type TripStatusLine =
  | {
      kind: "row";
      row: StatusRow;
      /** the fishery's own status, in Today's own word */
      word: string;
      level: TagLevel;
      /** the sentence after the tag */
      sentence: string;
      /** a registry notice for this species here that an agency release appears to lift, or whose sources differ */
      unclear: boolean;
    }
  | { kind: "tally"; tally: string; rows: StatusRow[] }
  | { kind: "none" };

/**
 * `rows`: Today's model rows for this port (null when the page has no port, so no curated record applies).
 * `unclearSpecies`: a notice for this species here is disputed (an agency release appears to lift it), so
 * the row is said as unclear, the words the rest of the app uses, instead of as closed or in effect.
 */
export function tripStatusLine(rows: StatusRow[] | null, species: Species, audience: Audience, unclearSpecies = false): TripStatusLine {
  if (!rows?.length) return { kind: "none" };
  if (species.id === "any") return { kind: "tally", tally: statusTally(rows), rows };
  const f = fisheryForTrip(species, audience);
  const row = f ? (rows.find((r) => r.fishery === f) ?? null) : null;
  if (!row) return { kind: "none" };
  const unclear = unclearSpecies || row.conflict !== null;
  if (unclear) return { kind: "row", row, word: "Status unclear", level: "watch", sentence: "awaiting a person's review. Read the agency's own page before you act.", unclear: true };
  const noRecord = row.level === "no_restriction_on_record";
  return {
    kind: "row",
    row,
    word: noRecord ? "None on record" : row.word,
    level: tagLevel(row),
    sentence: noRecord ? "Not a promise it is open." : row.headline,
    unclear: false,
  };
}
