import { portById } from "./places";

/** Suggested questions for a port: the common questions the deterministic router answers. */
export function suggestionsFor(placeId: string | null | undefined): string[] {
  const p = portById(placeId);
  const crab = "What's the status of Dungeness crab?";
  return [
    crab,
    "What's the latest crab toxin test?",
    "Sea, wind and tide tomorrow?",
    "What's the bloom outlook this week?",
    "What are the whale rules?",
    "What changed since I last checked?",
    p ? `Where does the ${p.label} data come from?` : "Where does this data come from?",
  ];
}
