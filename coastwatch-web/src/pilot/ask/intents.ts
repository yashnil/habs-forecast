import type { Fishery, Intent, SafetyKind } from "./types";

/**
 * Deterministic intent routing for the common questions. The same router runs in live mode first;
 * only questions it cannot place go to the (Haiku-class) model router.
 */

/* ------------------------------------------------------------------ safety (always first) */
const EAT =
  /\b(safe|ok|okay|alright|fine|good|healthy)\b[^.?!]{0,40}\b(eat|eating|consume|serve|cook|feed)\b|\b(can|could|should|may)\s+(i|we|you|my \w+)\s+(still\s+)?(eat|consume|serve)\b|\b(eat|eating)\b[^.?!]{0,30}\b(safe|ok|okay|risk|danger|sick|poison)|\bedible\b|\btoxic to eat\b|\b(safe|ok|okay) to (harvest|keep and eat|gather|collect)\b/i;
const BAR = /\b(cross(ing)?\s+the\s+bar|bar\s+(crossing|conditions|closed|closure|restriction\w*|report)|harbou?r\s+entrance)\b/i;
const GO_OUT =
  /\b(safe|ok|okay|good|smart|wise)\b[^.?!]{0,30}\b(go(ing)?\s+out|head(ing)?\s+out|launch|run\s+out|take\s+the\s+boat|be\s+on\s+the\s+water|fish(ing)?\s+tomorrow|go\s+fishing)\b|\bshould\s+(i|we)\s+(go|head|run)\s+(out|offshore)\b|\bis\s+it\s+safe\s+(out|offshore|on the water)\b|\bgo\s*\/\s*no[- ]go\b|\bsafe\s+(seas|conditions|day)\b/i;
const FISH_LEGAL =
  /\b(open|allowed)\s+so\s+(i|we)\s+can\b|\b(can|may)\s+(i|we)\s+(legally\s+)?(go\s+)?(fish|crab|set|drop|pull|keep|catch|sell)\b|\b(am|are)\s+(i|we)\s+allowed\b|\blegal(ly)?\s+to\s+(fish|crab|catch|keep|set|sell)\b|\b(ok|okay)\s+to\s+(fish|crab|set|drop)\b/i;

export function safetyKind(q: string): SafetyKind | null {
  if (EAT.test(q)) return "eat";
  if (BAR.test(q)) return "bar";
  if (GO_OUT.test(q)) return "go_out";
  if (FISH_LEGAL.test(q)) return "fish_legal";
  return null;
}

/* ---------------------------------------------------------------------------- fisheries */
const FISHERIES: [Fishery, RegExp][] = [
  ["rock_crab", /\brock\s*crab/i],
  ["dungeness", /\bdungeness|\bdungies?\b|\bcrab(s|bing|ber)?\b/i],
  ["salmon", /\bsalmon|\bchinook|\bking\s+salmon/i],
  ["groundfish", /\bground\s*fish|\brock\s*fish|\blingcod|\bsablefish|\bcabezon/i],
  ["anchovy_sardine", /\banchov|\bsardine/i],
  ["squid", /\bsquid/i],
  ["shellfish", /\bmussel|\bclam|\boyster|\bscallop|\bbivalve|\bshellfish/i],
  ["lobster", /\blobster/i],
];

export function fisheryIn(q: string): Fishery | null {
  for (const [f, re] of FISHERIES) if (re.test(q)) return f;
  return null;
}

/** Words that identify a fishery inside a PortFact's id/headline. */
export const FISHERY_WORDS: Record<Fishery, RegExp> = {
  dungeness: /dungeness/i,
  rock_crab: /rock[\s_-]*crab/i,
  salmon: /salmon/i,
  groundfish: /groundfish|rockfish|lingcod/i,
  anchovy_sardine: /anchov|sardine/i,
  squid: /squid/i,
  shellfish: /mussel|clam|oyster|scallop|bivalve|shellfish/i,
  lobster: /lobster/i,
};

export const FISHERY_LABEL: Record<Fishery, string> = {
  dungeness: "Dungeness crab",
  rock_crab: "rock crab",
  salmon: "salmon",
  groundfish: "groundfish",
  anchovy_sardine: "anchovy and sardine",
  squid: "market squid",
  shellfish: "shellfish",
  lobster: "spiny lobster",
};

/* ------------------------------------------------------------------------------ intents */
const RULES: [Intent, RegExp][] = [
  ["help", /^\s*(help|hi|hello|hey|info|what can (you|i) (do|ask)|how does this work)\b/i],
  ["sources", /\bwhere\s+(does|do|did)\b[^?]*\b(come|from)\b|\bsources?\b|\bhow do you know\b|\bwho (says|decides)\b|\bprovenance\b|\bdata from\b/i],
  ["data_age", /\bhow old\b|\bup to date\b|\bfresh\b|\bstale\b|\blast (updated|refresh\w*)\b|\bwhen\b[^?]*\b(updated|refreshed)\b/i],
  ["changed", /\bwhat('?s| has| is)?\s+(changed|new)\b|\bchanged\b|\banything new\b|\bsince (yesterday|last|my last|i last)\b|\bupdates?\b/i],
  ["whale", /\bwhales?\b|\bentangle\w*|\bramp\b|\bhumpback|\bdepth (limit|restriction)|\bgear reduction|\bfathom/i],
  ["toxin", /\btoxin|\bdomoic|\bbiotoxin|\bpsp\b|\bparalytic|\bppm\b|\btest(s|ed|ing)?\b|\blab\b|\bsample/i],
  ["bloom", /\bbloom|\balgae|\balgal|\bc-?harm\b|\bpseudo-?nitzschia|\bchlorophyll|\bred tide|\bhabs?\b|\boutlook\b/i],
  ["sea", /\bseas?\b|\bswell|\bwaves?\b|\bwind|\btides?\b|\bweather|\bbuoy|\bsurf\b|\bsmall craft|\bknots?\b|\bkt\b|\bfog\b|\bwater temp|\bsst\b|\btemperature|\bforecast\b|\bconditions\b/i],
  ["status", /\bopen(ed|er|ing)?\b|\bclosed?\b|\bclosures?\b|\bseason\b|\bstatus\b|\bdelay\w*|\brestrict\w*|\bfishery\b|\bquarantine|\bregulations?\b|\brules?\b|\btrip limits?\b/i],
];

export function routeIntent(q: string): Intent {
  for (const [intent, re] of RULES) if (re.test(q)) return intent;
  // "how's the crab?" — a fishery with no other cue is a status question
  if (fisheryIn(q)) return "status";
  return "unknown";
}

/** "tomorrow" / "today" / none — which forecast day a sea question is about. */
export function dayIn(q: string): "today" | "tomorrow" | null {
  if (/\btomorrow|\btmrw|\bmorning\b.*\bnext\b|\bnext day\b/i.test(q)) return "tomorrow";
  if (/\btoday|\bnow\b|\bright now|\bcurrently|\bthis (morning|afternoon)/i.test(q)) return "today";
  return null;
}
