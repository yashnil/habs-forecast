import type { SeenSnapshot } from "../types";

/**
 * What Ask last told this device about a port (localStorage "nx-ask-seen-v1:<port>"), so "what changed since
 * I last checked?" has something to compare with. It stays in the browser; storage that is blocked simply
 * compares with nothing.
 */
const SEEN_KEY = (place: string) => `nx-ask-seen-v1:${place}`;

export function readSeen(place: string): SeenSnapshot | null {
  try {
    const raw = localStorage.getItem(SEEN_KEY(place));
    return raw ? (JSON.parse(raw) as SeenSnapshot) : null;
  } catch {
    return null;
  }
}

export function writeSeen(place: string, s: SeenSnapshot) {
  try {
    localStorage.setItem(SEEN_KEY(place), JSON.stringify(s));
  } catch {
    /* storage unavailable: "what changed" just compares nothing */
  }
}
