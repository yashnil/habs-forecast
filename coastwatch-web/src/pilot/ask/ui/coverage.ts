import { PILOT_PORTS } from "@/pilot/contracts";
import { isPortKey } from "@/lib/portRecord";

/**
 * Which port Ask Neritic opens on. Ask answers only for the pilot ports. A reader who arrives from any other
 * port is told so and chooses a pilot port themselves: Ask never opens on a different port from the one that
 * was asked for and answers as though it were theirs.
 */
export type AskPort = { key: string; name: string | null };

export type AskCoverage =
  /** a pilot port was asked for: open on it */
  | { state: "covered"; place: string }
  /** a real port outside the pilot was asked for: say so, answer nothing until the reader picks a pilot port */
  | { state: "uncovered"; port: AskPort }
  /** no port was asked for: open on the first pilot port (the reader's saved port may replace it) */
  | { state: "open"; place: string };

const isPilot = (key: string | null | undefined): key is string => PILOT_PORTS.some((p) => p.place_id === key);

export function askCoverage(asked: string | null | undefined, names: Record<string, string> = {}): AskCoverage {
  if (isPilot(asked)) return { state: "covered", place: asked };
  if (isPortKey(asked)) return { state: "uncovered", port: { key: asked, name: names[asked] ?? null } };
  return { state: "open", place: PILOT_PORTS[0].place_id };
}

/** The same rule for the port saved on this device, when the address named none. */
export function savedCoverage(saved: string | null | undefined, names: Record<string, string> = {}): AskCoverage | null {
  if (!saved) return null;
  const c = askCoverage(saved, names);
  return c.state === "open" ? null : c;
}

export const PILOT_PORT_WORDS = (() => {
  const l = PILOT_PORTS.map((p) => p.label);
  return `${l.slice(0, -1).join(", ")} and ${l[l.length - 1]}`;
})();

/** What the page says for a port outside the pilot. */
export function uncoveredWords(port: AskPort): { title: string; body: string } {
  const name = port.name ?? "this port";
  return {
    title: `Ask Neritic does not cover ${name} yet.`,
    body: `During the pilot it answers only for ${PILOT_PORT_WORDS}. Nothing below is about ${name} until you choose one of those ports.`,
  };
}
