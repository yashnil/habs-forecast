import type { SpeciesId } from "@/content/species";

/**
 * What the evidence can and cannot say for one species: which marine biotoxins the agencies test
 * it for, which part of the animal matters, and which of Neritic's readings bear on that toxin.
 *
 * Every fact is the agency's own sentence, quoted with its page and the date it was read; `text` is
 * Neritic's short paraphrase for the API and is never shown as the agency's words. Nothing here is
 * Neritic's judgement of a species' safety, and no fact carries a date range that could read as an end
 * (notices with status and expected ends belong to the official registry). Pages read on 2026-10-16 and
 * every quote re-checked verbatim on 2026-10-18 (HTTP 200):
 * - CDPH Food and Drug Branch, "Domoic Acid" FAQ (crab and lobster);
 * - CDFW, "Ocean Fishing Health Advisories" (CDPH's shellfish and seawater monitoring).
 * The toxins an agency has acted on for a species are also read from the official records'
 * own `toxins` field (lib/harborEvidence), so a new notice can add to this list without a code change.
 */
export type Toxin = "domoic_acid" | "psp";

export const TOXIN_WORDS: Record<Toxin, string> = { domoic_acid: "domoic acid", psp: "paralytic shellfish toxins (saxitoxins)" };

/** Neritic's readings and what they are evidence of: Pseudo-nitzschia and particulate domoic acid in seawater, C-HARM's domoic acid probability */
export const READINGS_FOR: Record<Toxin, "carried" | "not-carried"> = { domoic_acid: "carried", psp: "not-carried" };

export type AgencyFact = { text: string; quote: string; source: { label: string; url: string; agency: "CDPH" | "CDFW" }; read: string };

const DA_FAQ = { agency: "CDPH" as const, label: "CDPH: domoic acid in crab and lobster", url: "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx" };
const CDFW_ADVISORIES = { agency: "CDFW" as const, label: "CDFW ocean fishing health advisories", url: "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" };
const READ = "2026-10-16";

const VISCERA: AgencyFact = {
  text: "The viscera (the “butter”) usually holds much more domoic acid than the meat.",
  quote: "CDPH recommends consumers not eat the viscera (internal organs, also known as \"butter\" or \"guts\") of crab, as the viscera usually contains much higher levels of domoic acid than the meat.",
  source: DA_FAQ,
  read: READ,
};
const BIVALVES: AgencyFact = {
  text: "CDPH tests mussels, clams and scallops along the coast for paralytic shellfish toxins and domoic acid.",
  quote: "The California Department of Public Health (CDPH) coordinates a routine monitoring program along the California coast to sample mussels and other shellfish like clams and scallops for the presence of Paralytic Shellfish Poisoning (PSP) and domoic acid toxins.",
  source: CDFW_ADVISORIES,
  read: READ,
};

export type SpeciesEvidence = {
  /** the biotoxins the agencies' pages say they test this species for */
  toxins: Toxin[];
  /** who tests the animal itself, in the agency's words */
  facts: AgencyFact[];
};

export const SPECIES_EVIDENCE: Partial<Record<SpeciesId, SpeciesEvidence>> = {
  dungeness: {
    toxins: ["domoic_acid"],
    facts: [
      {
        text: "CDPH tests Dungeness crab before the season opens.",
        quote: "CDPH conducts pre-season testing of Dungeness crab to ensure that the species are not accumulating dangerous levels of domoic acid.",
        source: DA_FAQ,
        read: READ,
      },
      VISCERA,
    ],
  },
  "rock-crab": { toxins: ["domoic_acid"], facts: [VISCERA] },
  lobster: {
    toxins: ["domoic_acid"],
    facts: [
      {
        text: "CDPH may test lobster when a bloom, or rising domoic acid in shellfish, is seen.",
        quote: "CDPH may increase sampling of small fin fish, crab, lobsters and other impacted species when algal blooms or increased levels of domoic acid in bi-valve shellfish are observed",
        source: DA_FAQ,
        read: READ,
      },
    ],
  },
  // The annual quarantine is not a species fact: it is an official notice with a status, an expected end that
  // CDPH can extend (SN26-018: "will continue through at least October 31") and a review state. The registry
  // record (cdph-…-annual-mussel-quarantine) speaks for it; a dated sentence here would read as "over" after
  // October 31 (#66 review 3.1).
  mussel: { toxins: ["psp", "domoic_acid"], facts: [BIVALVES] },
  bivalve: { toxins: ["psp", "domoic_acid"], facts: [BIVALVES] },
  "razor-clam": { toxins: ["psp", "domoic_acid"], facts: [BIVALVES] },
  "anchovy-sardine": {
    toxins: ["domoic_acid"],
    facts: [
      {
        text: "CDPH may test small fin fish when a bloom, or rising domoic acid in shellfish, is seen.",
        quote: "CDPH may increase sampling of small fin fish, crab, lobsters and other impacted species when algal blooms or increased levels of domoic acid in bi-valve shellfish are observed",
        source: DA_FAQ,
        read: READ,
      },
    ],
  },
};
