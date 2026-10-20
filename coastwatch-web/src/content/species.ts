/**
 * Target species for My Coast's "Plan" view (Coastal mode).
 *
 * `match` words are compared with each official record's *species* field, which is the agency's
 * own list of species (e.g. "razor clam", "northern anchovy", "rock crab (internal organs)").
 * They never match place names. A notice with no species overlap is still listed under
 * "other notices for these waters": choosing a species re-orders, it never hides.
 *
 * `statementTopic` links CDFW's statewide statements (e.g. "no closures in the Dungeness crab
 * fishery due to naturally occurring marine toxins"), quoted verbatim where shown.
 *
 * Links were checked on 2026-10-13 (HTTP 200).
 */
export type SpeciesId =
  | "any"
  | "dungeness"
  | "rock-crab"
  | "razor-clam"
  | "mussel"
  | "bivalve"
  | "anchovy-sardine"
  | "salmon"
  | "groundfish"
  | "halibut"
  | "squid"
  | "lobster"
  | "tunas"
  | "urchin";

export type Species = {
  id: SpeciesId;
  label: string;
  /** words in a record's species list that make it about this species */
  match: string[];
  activities: ("commercial" | "recreational" | "shellfish")[];
  statementTopic?: string;
  /** the agency page for this species' rules and status */
  link?: { label: string; url: string };
};

const CDFW_INVERT_REGS = { label: "CDFW invertebrate fishing regulations", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Sport-Fishing/Invertebrate-Fishing-Regs" };

export const SPECIES: Species[] = [
  { id: "any", label: "Any species", match: [], activities: ["commercial", "recreational", "shellfish"] },
  {
    id: "dungeness",
    label: "Dungeness crab",
    match: ["dungeness"],
    activities: ["commercial", "recreational"],
    statementTopic: "dungeness",
    link: { label: "CDFW Whale Safe Fisheries: Dungeness crab season status", url: "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries" },
  },
  { id: "rock-crab", label: "Rock crab", match: ["rock crab"], activities: ["commercial", "recreational"], link: { label: "CDFW rock crab", url: "https://wildlife.ca.gov/Regions/Marine/Rock-Crab" } },
  { id: "razor-clam", label: "Razor clams", match: ["razor"], activities: ["shellfish", "recreational"], link: { label: "CDFW razor clams", url: "https://wildlife.ca.gov/Conservation/Marine/Invertebrates/Razor-Clams" } },
  { id: "mussel", label: "Mussels", match: ["mussel", "bivalve"], activities: ["shellfish", "recreational"], link: CDFW_INVERT_REGS },
  { id: "bivalve", label: "Clams, oysters, scallops", match: ["clam", "oyster", "scallop", "bivalve"], activities: ["shellfish", "recreational"], link: CDFW_INVERT_REGS },
  {
    id: "anchovy-sardine",
    label: "Anchovy or sardine",
    match: ["anchov", "sardine"],
    activities: ["commercial", "recreational"],
    link: { label: "CDFW pelagic fisheries", url: "https://wildlife.ca.gov/Conservation/Marine/Pelagic-Fisheries" },
  },
  { id: "salmon", label: "Salmon", match: ["salmon"], activities: ["commercial", "recreational"], link: { label: "CDFW ocean salmon regulations", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Salmon" } },
  {
    id: "groundfish",
    label: "Rockfish and groundfish",
    match: ["rockfish", "groundfish", "lingcod", "sablefish"],
    activities: ["commercial", "recreational"],
    link: { label: "CDFW groundfish regulations summary", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Groundfish-Summary" },
  },
  { id: "halibut", label: "California halibut", match: ["halibut"], activities: ["commercial", "recreational"], link: { label: "CDFW California halibut", url: "https://wildlife.ca.gov/Regions/Marine/California-Halibut" } },
  { id: "squid", label: "Market squid", match: ["squid"], activities: ["commercial"], link: { label: "CDFW market squid", url: "https://wildlife.ca.gov/Conservation/Marine/Pelagic-Fisheries/Market-Squid" } },
  { id: "lobster", label: "Spiny lobster", match: ["lobster"], activities: ["commercial", "recreational"], statementTopic: "lobster", link: { label: "CDFW spiny lobster", url: "https://wildlife.ca.gov/Regions/Marine/Lobster" } },
  // v5 (fishermen audit): the offshore and dive fisheries people asked about; links checked 2026-10-20 (HTTP 200)
  { id: "tunas", label: "Albacore and other tunas", match: ["tuna", "albacore"], activities: ["commercial", "recreational"], link: { label: "CDFW ocean sport fishing regulations", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations" } },
  { id: "urchin", label: "Red sea urchin", match: ["urchin"], activities: ["commercial"], link: { label: "CDFW sea urchin", url: "https://wildlife.ca.gov/Conservation/Marine/Invertebrates/Sea-Urchin" } },
];

/** The label inside a sentence: "razor clams", but "Dungeness crab" and "California halibut" keep their capital. */
export function speciesPhrase(s: Species): string {
  return /^(Dungeness|California)/.test(s.label) ? s.label : s.label.charAt(0).toLowerCase() + s.label.slice(1);
}

export function speciesById(id: string | null | undefined): Species {
  return SPECIES.find((s) => s.id === id) ?? SPECIES[0];
}

/** Whether an official record names this species in its own species list. */
export function recordNamesSpecies(recordSpecies: string[], s: Species): boolean {
  if (s.id === "any") return true;
  const text = recordSpecies.join(" ").toLowerCase();
  return s.match.some((w) => text.includes(w));
}
