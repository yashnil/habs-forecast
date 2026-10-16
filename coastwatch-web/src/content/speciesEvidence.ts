import type { SpeciesId } from "@/content/species";

/**
 * What the evidence can and cannot say for one species: which marine biotoxins the agencies test
 * it for, which part of the animal matters, and which of Neritic's readings bear on that toxin.
 *
 * Every fact is the agency's own sentence, quoted with its page and the date it was read; nothing
 * here is Neritic's judgement of a species' safety. Pages read on 2026-10-16 (HTTP 200):
 * - CDPH Food and Drug Branch, "Domoic Acid" FAQ (crab and lobster);
 * - CDPH Shellfish Program, "Marine Biotoxin Monitoring Program" (bivalves; the mussel quarantine);
 * - CDFW, "Ocean Fishing Health Advisories" (CDPH's shellfish and seawater monitoring).
 * The toxins an agency has acted on for a species are also read from the official records'
 * own `toxins` field (lib/harborEvidence), so a new notice can add to this list without a code change.
 */
export type Toxin = "domoic_acid" | "psp";

export const TOXIN_WORDS: Record<Toxin, string> = { domoic_acid: "domoic acid", psp: "paralytic shellfish toxins (saxitoxins)" };

/** Neritic's readings and what they are evidence of: Pseudo-nitzschia and particulate domoic acid in seawater, C-HARM's domoic acid probability */
export const READINGS_FOR: Record<Toxin, "carried" | "not-carried"> = { domoic_acid: "carried", psp: "not-carried" };

export type AgencyFact = { text: string; quote: string; source: { label: string; url: string }; read: string };

const DA_FAQ = { label: "CDPH: domoic acid in crab and lobster", url: "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx" };
const BIOTOXIN = { label: "CDPH Marine Biotoxin Monitoring Program", url: "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx" };
const CDFW_ADVISORIES = { label: "CDFW ocean fishing health advisories", url: "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" };
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
  mussel: {
    toxins: ["psp", "domoic_acid"],
    facts: [
      {
        text: "CDPH quarantines sport-harvested mussels from May 1 to October 31 every year.",
        quote: "CDPH issues a mussel quarantine from May 1 to October 31 each year.",
        source: BIOTOXIN,
        read: READ,
      },
      BIVALVES,
    ],
  },
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
