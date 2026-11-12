/**
 * Everything Neritic keeps in this browser, in one list: each localStorage and sessionStorage key (or key
 * prefix) and each Cache Storage cache, with what it holds in the reader's words. The account page's
 * inventory, the export and "Delete everything on this device" (forget.ts) read this list, and a unit test
 * (tests/unit/nx/personal/keys.test.ts) fails when a source file names a storage key that is not here, so a
 * new key cannot be kept without being listed, exported and deleted.
 *
 * Pure data and string matching: no window, no imports (the registry is read by tests and by the server).
 */
export type KeyStore = "local" | "session" | "cache";

/**
 * What a key is, for the inventory's grouping:
 *   personal    the reader's own choices and records (ports, fisheries, trips, alerts, account)
 *   history     what this device saw before (snapshots a "what changed" line or an alert compares with)
 *   offline     pages and files saved for use without a connection
 *   preference  how the app looks or behaves (chart mode, tours dismissed)
 *   transient   housekeeping with no meaning to the reader (probes, animation positions, device capability)
 */
export type KeyKind = "personal" | "history" | "offline" | "preference" | "transient";

export type KeyEntry = {
  /** the exact key, or the prefix every such key starts with */
  key: string;
  /** "prefix": keys are `key` + something (a port, a pack id) */
  match: "exact" | "prefix";
  store: KeyStore;
  kind: KeyKind;
  /** what it holds, in the reader's words (the inventory prints this) */
  holds: string;
  /** an older build wrote it; kept readable so a rollback still finds it */
  legacy?: true;
};

const local = (key: string, kind: KeyKind, holds: string, o: Partial<KeyEntry> = {}): KeyEntry => ({ key, match: key.endsWith(":") || key.endsWith("-") ? "prefix" : "exact", store: "local", kind, holds, ...o });
const session = (key: string, holds: string, o: Partial<KeyEntry> = {}): KeyEntry => ({ ...local(key, "transient", holds, o), store: "session" });
const cache = (key: string, kind: KeyKind, holds: string, o: Partial<KeyEntry> = {}): KeyEntry => ({ ...local(key, kind, holds, o), store: "cache" });

export const KEYS: KeyEntry[] = [
  // ---------------------------------------------------------------- the reader's port, fisheries and follows
  local("nx-port-v1", "personal", "Your home port, the port you opened last, your fisheries in order and your vessel length"),
  local("nx-follow-v1", "personal", "The ports you follow and what should alert you at each"),
  local("nx-coast-v1", "personal", "The fishery you last planned for on My Coast"),
  // ---------------------------------------------------------------- trips and vessel economics (My Coast)
  local("nx-trip-v1:", "personal", "Your saved trips, vessels and their costs and prices (My Coast's worksheet)"),
  local("cw-trip-checks-v1:", "personal", "Trip checklist ticks from an earlier version", { legacy: true }),
  local("cw-sea-limits-v1", "personal", "Your own wind and sea limits from an earlier version", { legacy: true }),
  // ---------------------------------------------------------------- alerts on this device
  local("nx-inbox-v1", "personal", "Your alerts: what changed at the ports you follow, and which you have read"),
  local("nx-alert-snap-v1", "history", "What each followed port's records said at the last check (alerts compare with it)"),
  local("nx-alert-check-v1", "history", "When this device last checked the ports you follow"),
  // ---------------------------------------------------------------- what this device saw before
  local("nx-today-seen-v1:", "history", "What Today showed at each port on your last visit (the \"since your last visit\" line compares with it)"),
  local("nx-ask-seen-v1:", "history", "What Ask Neritic last told you about a port (so it can say what changed)"),
  session("nx-today-visit-v1:", "The earlier visit this browser tab is comparing Today with"),
  // ---------------------------------------------------------------- offline
  local("nx-field-v1", "offline", "Which port is saved for offline use, when, and what the copy holds"),
  local("nx-today-saved-v1", "offline", "Which ports' Today pages are saved for offline use, and when"),
  cache("neritic-pack-", "offline", "Pages, records and chart pictures saved for use without a connection"),
  local("neritic-pack-saved-", "offline", "When an offline pack from an earlier version was saved", { legacy: true }),
  cache("neritic-kept-v1", "offline", "Briefings kept for offline use in an earlier version", { legacy: true }),
  local("cw-kept-v1", "offline", "The list of briefings kept in an earlier version", { legacy: true }),
  // ---------------------------------------------------------------- preferences
  local("nx-chart-mode-v1", "preference", "The chart's day, dusk or night mode"),
  local("nx-tour-v1", "preference", "That you dismissed \"Show me around\""),
  local("nx-opening-seen-v1", "preference", "That the opening has played once"),
  local("nx-currents-view-v1", "preference", "How Explore draws currents"),
  local("nx-explore-coach-v1", "preference", "That Explore's first hint was dismissed"),
  local("nx-a2hs-hint-v1", "preference", "That the home-screen hint was dismissed"),
  local("nx-daylight-v1", "preference", "The bright-daylight theme of an earlier version", { legacy: true }),
  local("nx-motion-force-v1", "preference", "A motion override used for testing"),
  local("nx-splash-seen-v2", "preference", "That an earlier version's opening has played", { legacy: true }),
  local("nx-splash-off-v1", "preference", "That an earlier version's opening is switched off", { legacy: true }),
  local("nx-splash-force-v1", "preference", "An opening override used for testing", { legacy: true }),
  local("cw-mode-v1", "preference", "Coastal or Research presentation in an earlier version", { legacy: true }),
  local("cw-purpose-v1", "preference", "What you said you use Neritic for in an earlier version", { legacy: true }),
  local("cw-intro-dismissed-v1", "preference", "That an earlier version's introduction was dismissed", { legacy: true }),
  local("cw-probability-classes-v1", "preference", "How the map classed probabilities in an earlier version", { legacy: true }),
  local("cw-chlorophyll-log-v1", "preference", "The chlorophyll colour scale of an earlier version", { legacy: true }),
  local("neritic-sst-thermal-v1", "preference", "The temperature colour scale of an earlier version", { legacy: true }),
  // ---------------------------------------------------------------- the pilot account's device copy
  local("nrt-pilot-account-v1", "personal", "The pilot account's copy on this device: role, saved ports, alert choices and consents"),
  local("nrt-pilot-inbox-v1", "personal", "The pilot account's alert inbox on this device"),
  local("nrt-pilot-alert-snap-v1", "history", "What the pilot ports' records said at the pilot account's last check"),
  local("nrt-pilot-feedback-v1", "personal", "Your \"Was this useful?\" answers"),
  local("nrt-pilot-trips-v1", "personal", "Trip feedback you wrote for the pilot"),
  // ---------------------------------------------------------------- older ports and waters
  local("cw-home-port", "personal", "Your home port from an earlier version", { legacy: true }),
  local("cw-my-waters", "personal", "Your saved waters from an earlier version", { legacy: true }),
  local("cw-waters-v1", "personal", "Your saved waters from an earlier version", { legacy: true }),
  local("nx-today-place-v1", "personal", "The port you opened last, from an earlier version", { legacy: true }),
  local("nx-today-fisheries-v1", "personal", "Your fisheries from an earlier version", { legacy: true }),
  // ---------------------------------------------------------------- housekeeping
  local("nx-today-probe", "transient", "A test that this browser keeps data"),
  local("nrt-probe", "transient", "A test that this browser keeps data"),
  local("cw-probe", "transient", "A test that this browser keeps data"),
  local("nx-gfx-losses-v1", "transient", "How often this device's graphics failed (the chart then draws a still picture)"),
  local("nx-gfx-slow-v1", "transient", "That this device draws the chart slowly"),
  session("nx-nav-pill-v1", "Where the navigation highlight last sat"),
  session("nx-stage-arrived:", "That an earlier version's chart has arrived at a place"),
  session("cw-glide-", "Where an animated highlight last sat"),
  session("cw-camera-tier-v1", "How far this device's camera moves travel"),
  session("cw-changes-session-v1", "What an earlier version's \"what changed\" compared with", { legacy: true }),
  // the app's own code for offline use: nothing about the reader, replaced on every release
  cache("neritic-static-", "transient", "Neritic's own code and type, so saved pages open offline"),
];

/** The entry a key or cache name belongs to, or null when it is not Neritic's (or not registered). */
export function entryFor(name: string, store?: KeyStore): KeyEntry | null {
  let best: KeyEntry | null = null;
  for (const e of KEYS) {
    if (store && e.store !== store) continue;
    const hit = e.match === "exact" ? name === e.key : name.startsWith(e.key);
    // the longest match wins: "neritic-pack-saved-x" is the saved mark, not a pack
    if (hit && (!best || e.key.length > best.key.length)) best = e;
  }
  return best;
}

export const isRegistered = (name: string, store?: KeyStore) => entryFor(name, store) !== null;

/**
 * Keys this origin holds that are not Neritic's own data and are never deleted or exported by forget.ts:
 * a signed-in account's session (Supabase's `sb-…` keys) is ended by signing out, not by clearing the device.
 */
export const FOREIGN_PREFIXES = ["sb-"] as const;
export const isForeign = (name: string) => FOREIGN_PREFIXES.some((p) => name.startsWith(p));

/**
 * Caches "Delete everything" leaves alone: the app's own code (`neritic-static-…`) says nothing about the
 * reader, and removing it would only make the next offline open fail.
 */
export const keepsOnForget = (cacheName: string) => cacheName.startsWith("neritic-static-");

export const KIND_ORDER: KeyKind[] = ["personal", "history", "offline", "preference", "transient"];
export const KIND_WORDS: Record<KeyKind, string> = {
  personal: "Your ports, fisheries, trips and alerts",
  history: "What this device saw before",
  offline: "Saved for offline use",
  preference: "How Neritic looks and behaves",
  transient: "Housekeeping",
};
