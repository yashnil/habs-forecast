import { PILOT_PORTS } from "../contracts";

/** Resolving a place from a question, a URL parameter or an SMS keyword. Pilot ports only. */

export type PilotPort = (typeof PILOT_PORTS)[number];

const ALIASES: { re: RegExp; place_id: string }[] = [
  { re: /\bmoss(\s+landing)?\b|\bmoss\b/i, place_id: "port-592" },
  { re: /\bsanta\s*cruz\b|\bsantacruz\b/i, place_id: "port-593" },
  { re: /\bhalf\s*moon\s*bay\b|\bhmb\b|\bprinceton\b|\bpillar\s*point\b/i, place_id: "port-452" },
  // "Monterey Bay" alone names the bay, not the harbor; still the Monterey port's records
  { re: /\bmonterey\b|\bmry\b/i, place_id: "port-550" },
];

/** Places outside the pilot (Ask Neritic says so instead of guessing). */
const OUT_OF_AREA =
  /\b(crescent\s*city|eureka|trinidad|humboldt|fort\s*bragg|noyo|shelter\s*cove|mendocino|bodega|tomales|san\s*francisco|sf\s*bay|oakland|richmond|sausalito|point\s*reyes|morro\s*bay|avila|port\s*san\s*luis|san\s*luis\s*obispo|santa\s*barbara|ventura|oxnard|channel\s*islands|san\s*pedro|los\s*angeles|long\s*beach|newport\s*beach|dana\s*point|oceanside|san\s*diego|oregon|washington|seattle|astoria|alaska|hawaii|mexico|baja|florida|maine|gulf)\b/i;

export function portById(id: string | null | undefined): PilotPort | null {
  return PILOT_PORTS.find((p) => p.place_id === id) ?? null;
}

/** A pilot port named in free text, or null. */
export function portInText(text: string): PilotPort | null {
  for (const a of ALIASES) if (a.re.test(text)) return portById(a.place_id);
  return null;
}

/** A named place outside the pilot ports (only when no pilot port is also named). */
export function outOfAreaInText(text: string): string | null {
  const m = text.match(OUT_OF_AREA);
  return m ? m[0] : null;
}

export type PlaceResolution =
  | { kind: "port"; port: PilotPort }
  | { kind: "out_of_area"; named: string }
  | { kind: "none" };

/** The question's own place wins over the selected one; out-of-area names are never guessed. */
export function resolvePlace(question: string, selected?: string | null): PlaceResolution {
  const named = portInText(question);
  if (named) return { kind: "port", port: named };
  const away = outOfAreaInText(question);
  if (away) return { kind: "out_of_area", named: away };
  const sel = portById(selected);
  return sel ? { kind: "port", port: sel } : { kind: "none" };
}

export const PILOT_PORT_NAMES = PILOT_PORTS.map((p) => p.label).join(", ");
