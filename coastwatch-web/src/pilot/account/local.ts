import { PILOT_PORTS, type FisheryId } from "@/pilot/contracts";
import { TRIGGERS, type AlertTrigger } from "@/pilot/alerts/diff";
import { readJSON, removeKeys, writeJSON } from "./storage";

/**
 * The pilot account as kept on this device. With accounts not yet connected this IS the account;
 * once Supabase is configured and the person signs in, it is synced to their rows (sync.ts) and
 * stays as the offline copy. Ports and fisheries are the pilot's own ids (contracts.ts).
 */
export const ACCOUNT_KEY = "nrt-pilot-account-v1";

export const FISHERIES: { id: FisheryId; label: string }[] = [
  { id: "dungeness_commercial", label: "Dungeness crab (commercial)" },
  { id: "dungeness_recreational", label: "Dungeness crab (recreational)" },
  { id: "rock_crab", label: "Rock crab" },
  { id: "salmon_commercial", label: "Salmon (commercial)" },
  { id: "groundfish", label: "Groundfish / rockfish" },
  { id: "anchovy_sardine", label: "Anchovy and sardine" },
  { id: "market_squid", label: "Market squid" },
  { id: "bivalve_shellfish", label: "Mussels, clams and oysters" },
  { id: "spiny_lobster", label: "Spiny lobster" },
];

export const ROLES = [
  { id: "commercial_fisher", label: "Commercial fisherman" },
  { id: "recreational_fisher", label: "Recreational angler" },
  { id: "charter", label: "Charter / CPFV" },
  { id: "processor", label: "Processor or buyer" },
  { id: "scientist", label: "Scientist" },
  { id: "agency", label: "Agency staff" },
  { id: "other", label: "Other" },
] as const;
export type Role = (typeof ROLES)[number]["id"];

export const VESSEL_CLASSES = [
  { id: "none", label: "No vessel / shore" },
  { id: "under_25ft", label: "Under 25 ft" },
  { id: "25_40ft", label: "25–40 ft" },
  { id: "40_60ft", label: "40–60 ft" },
  { id: "over_60ft", label: "Over 60 ft" },
] as const;
export type VesselClass = (typeof VESSEL_CLASSES)[number]["id"];

export type Channel = "in_app" | "web_push" | "sms";

export type AlertPref = { place_id: string; triggers: AlertTrigger[]; channels: Channel[] };

/** A consent the person gave: when, and the exact words shown. */
export type ConsentRecord = { at: string; text: string };

export type LocalAccount = {
  v: 1;
  role: Role | null;
  fisheries: FisheryId[];
  home_place_id: string | null;
  vessel_length_class: VesselClass | null;
  /** 13+ attestation (required before anything is kept) */
  age_13_plus: ConsentRecord | null;
  /** saved ports, in the order saved */
  saved_places: string[];
  alerts: AlertPref[];
  sms: { phone_e164: string; consent: ConsentRecord } | null;
  created_at: string | null;
};

export const EMPTY_ACCOUNT: LocalAccount = {
  v: 1,
  role: null,
  fisheries: [],
  home_place_id: null,
  vessel_length_class: null,
  age_13_plus: null,
  saved_places: [],
  alerts: [],
  sms: null,
  created_at: null,
};

export const MAX_SAVED = 8;

const isAccount = (v: unknown): v is LocalAccount => !!v && typeof v === "object" && (v as LocalAccount).v === 1 && Array.isArray((v as LocalAccount).saved_places);

/** Read tolerantly: unknown fisheries, triggers or malformed parts are dropped, never trusted. */
export function readAccount(): LocalAccount {
  const a = readJSON<LocalAccount>(ACCOUNT_KEY, EMPTY_ACCOUNT, isAccount);
  const fish = new Set(FISHERIES.map((f) => f.id));
  const trig = new Set<string>(TRIGGERS);
  return {
    ...EMPTY_ACCOUNT,
    ...a,
    fisheries: (a.fisheries ?? []).filter((f) => fish.has(f)),
    saved_places: (a.saved_places ?? []).filter((p) => typeof p === "string" && /^[a-z0-9:-]{1,64}$/.test(p)).slice(0, MAX_SAVED),
    alerts: (Array.isArray(a.alerts) ? a.alerts : [])
      .filter((p) => p && typeof p.place_id === "string")
      .map((p) => ({ place_id: p.place_id, triggers: (p.triggers ?? []).filter((t) => trig.has(t)), channels: (p.channels ?? []).filter((c) => c === "in_app" || c === "web_push" || c === "sms") })),
  };
}

export function writeAccount(a: LocalAccount): boolean {
  return writeJSON(ACCOUNT_KEY, a);
}

export function updateAccount(fn: (a: LocalAccount) => LocalAccount): { account: LocalAccount; persisted: boolean } {
  const next = fn(readAccount());
  return { account: next, persisted: writeAccount(next) };
}

export const DEFAULT_TRIGGERS: AlertTrigger[] = ["status_change", "new_toxin_result", "ramp_action"];

export function savePlace(a: LocalAccount, place: string): LocalAccount {
  if (a.saved_places.includes(place)) return a;
  const saved = [...a.saved_places, place].slice(0, MAX_SAVED);
  // a newly saved port gets in-app alerts for official changes by default (the person can turn them off)
  const alerts = a.alerts.some((p) => p.place_id === place) ? a.alerts : [...a.alerts, { place_id: place, triggers: DEFAULT_TRIGGERS, channels: ["in_app" as Channel] }];
  return { ...a, saved_places: saved, alerts, home_place_id: a.home_place_id ?? place };
}

export function removePlace(a: LocalAccount, place: string): LocalAccount {
  return {
    ...a,
    saved_places: a.saved_places.filter((p) => p !== place),
    alerts: a.alerts.filter((p) => p.place_id !== place),
    home_place_id: a.home_place_id === place ? (a.saved_places.find((p) => p !== place) ?? null) : a.home_place_id,
  };
}

export function setTrigger(a: LocalAccount, place: string, t: AlertTrigger, on: boolean): LocalAccount {
  const cur = a.alerts.find((p) => p.place_id === place) ?? { place_id: place, triggers: [], channels: ["in_app" as Channel] };
  const triggers = on ? [...new Set([...cur.triggers, t])] : cur.triggers.filter((x) => x !== t);
  const ordered = TRIGGERS.filter((x) => triggers.includes(x));
  return { ...a, alerts: [...a.alerts.filter((p) => p.place_id !== place), { ...cur, triggers: ordered }] };
}

export function setChannel(a: LocalAccount, place: string, c: Channel, on: boolean): LocalAccount {
  const cur = a.alerts.find((p) => p.place_id === place) ?? { place_id: place, triggers: DEFAULT_TRIGGERS, channels: [] as Channel[] };
  const channels = on ? [...new Set([...cur.channels, c])] : cur.channels.filter((x) => x !== c);
  return { ...a, alerts: [...a.alerts.filter((p) => p.place_id !== place), { ...cur, channels }] };
}

export function placeLabel(id: string): string {
  return PILOT_PORTS.find((p) => p.place_id === id)?.label ?? id;
}

/** Forget everything this feature kept on this device (account, inbox, snapshots, feedback, trips). */
export function forgetDevice(): void {
  removeKeys([ACCOUNT_KEY, "nrt-pilot-inbox-v1", "nrt-pilot-alert-snap-v1", "nrt-pilot-feedback-v1", "nrt-pilot-trips-v1"]);
}
