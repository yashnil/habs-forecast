/**
 * Today at your port: the facts (Neritic private pilot, track T).
 *
 * `buildPortToday(inputs)` is a PURE function. It turns data that was already loaded into the
 * statements Today shows, with stable ids (`PortFact.id`): the official status per fishery, CDPH toxin
 * tests, whale-entanglement (RAMP) rules, the C-HARM run, the 8-day chlorophyll outlook, sea and
 * weather, and data age. The data are the manifest's C-HARM run, `derived/index.json`, the biotoxin
 * and fisheries-status sidecars, official notices and `/api/conditions`. Alerts and Ask Neritic reuse
 * the same facts. `buildTodayModel` returns the same facts together with the structured views the
 * page draws (status rows, test sites, the bloom timeline, sea lanes).
 *
 * Rules (docs/coastwatch/05-science-and-safety.md; contracts.ts):
 * - Hierarchy: official status → measured tests and observations → forecasts → outlook → data age.
 *   An official closure or delay is always the first fact, and nothing lower down may contradict it.
 * - No fact ever calls anything "safe", "clear" or "go". The calmest status is "No restriction on
 *   record", which is not a guarantee.
 * - Chlorophyll is never toxin. The outlook always says "a bloom indicator, not a toxin forecast".
 *   C-HARM is a probability of conditions in the water, not toxin in seafood.
 * - Observations are never forecasts: buoy and harbour readings are "measured" with their own time.
 * - Curated official records stay unverified until a person reviews them (`verified: false`).
 * - Stale data is flagged in place, never silently shown as current.
 */
import type {
  BiotoxinIndex,
  BiotoxinSet,
  BiotoxinSite,
  FactKind,
  FisheriesStatusIndex,
  FisheryId,
  FisheryStatus,
  PortFact,
  PortToday,
  RampState,
  Severity,
  StatusLevel,
} from "./contracts";
import type { Conditions, Tide } from "@/lib/conditions";
import { compass } from "@/lib/conditions";
import { lanePeriods, periodLine, type LanePeriod } from "./today/marine";

// ---------------------------------------------------------------- inputs

export type TodayPlace = { place_id: string; label: string; lat: number; lon: number };

/** An official notice (CDFW/CDPH registry) already tied to this place by its official geography. */
export type NoticeInput = {
  id: string;
  agency: string;
  title: string;
  summary: string;
  action: "fishery_closure" | "take_restriction" | "consumption_advisory" | "quarantine" | "special_advisory" | "reopening" | "advisory_lifted";
  species: string[];
  fishery: string;
  effective: string | null;
  url: string | null;
  verified: boolean;
};

/** C-HARM at the port for one day: the port's own 3 km cell when it has a value, else the cells within 15 km. */
export type CharmProb = { here: number | null; median: number | null; max: number | null; n: number; nearestKm: number | null };
export type CharmDayInput = { lead: number; date: string | null; pda: CharmProb | null; pn: CharmProb | null };
export type CharmInput = {
  issued: string | null;
  /** lib/freshness state of the run */
  state: "current" | "stale" | "historical" | "unavailable";
  ageDays: number | null;
  days: CharmDayInput[];
  url: string;
};

/** The 8-day chlorophyll outlook (derived/index.json `chl_outlook`) at the port, with its skill as published. */
export type OutlookInput = {
  id: string;
  label: string;
  issued: string;
  valid_from: string;
  valid_to: string;
  /** mg m-3; the port's cell, or the median of water cells within 15 km (`basis`) */
  value: number | null;
  range: [number, number] | null;
  basis: "cell" | "nearby" | null;
  skill: { metric: string; model: number; [k: string]: unknown };
  caveats: string[];
  method: string;
};

export type PortTodayInputs = {
  place: TodayPlace;
  /** the clock, ISO (server: CW_FIXED_NOW in tests) */
  now: string;
  dataGeneratedAt: string | null;
  /** fisheries the reader follows (default: DEFAULT_FISHERIES) */
  selected?: FisheryId[];
  fisheries: FisheriesStatusIndex | null;
  biotoxin: BiotoxinIndex | null;
  /** null: the notices could not be loaded */
  notices: NoticeInput[] | null;
  charm: CharmInput | null;
  outlook: OutlookInput | null;
  /** null: not loaded yet (no sea facts); "error": could not be loaded */
  conditions: Conditions | null | "error";
};

// ---------------------------------------------------------------- vocabulary

export const FISHERY_OPTIONS: { id: FisheryId; label: string }[] = [
  { id: "dungeness_commercial", label: "Dungeness crab (commercial)" },
  { id: "dungeness_recreational", label: "Dungeness crab (recreational)" },
  { id: "rock_crab", label: "Rock crab" },
  { id: "salmon_commercial", label: "Salmon (commercial)" },
  { id: "groundfish", label: "Groundfish" },
  { id: "anchovy_sardine", label: "Anchovy & sardine" },
  { id: "market_squid", label: "Market squid" },
  { id: "bivalve_shellfish", label: "Shellfish (mussels, clams, oysters)" },
  { id: "spiny_lobster", label: "Spiny lobster" },
];
export const DEFAULT_FISHERIES: FisheryId[] = ["dungeness_commercial", "rock_crab", "salmon_commercial", "groundfish", "anchovy_sardine", "market_squid", "bivalve_shellfish"];
const FISHERY_IDS = new Set<string>(FISHERY_OPTIONS.map((f) => f.id));
export const fisheryLabel = (f: FisheryId) => FISHERY_OPTIONS.find((o) => o.id === f)?.label ?? f.replace(/_/g, " ");

/** "dungeness_commercial,groundfish" → valid ids in the reader's order; empty or invalid → null. */
export function parseFisheries(v: string | null | undefined): FisheryId[] | null {
  if (!v) return null;
  const ids = v.split(",").map((s) => s.trim()).filter((s, i, a) => FISHERY_IDS.has(s) && a.indexOf(s) === i) as FisheryId[];
  return ids.length ? ids : null;
}

export const LEVEL_WORD: Record<StatusLevel | "unknown", string> = {
  closed: "Closed",
  delayed: "Delayed",
  restricted: "Restricted",
  watch: "Watch",
  no_restriction_on_record: "No restriction on record",
  unknown: "Status not available",
};
export const CAUSE_WORD: Record<FisheryStatus["cause"], string> = {
  whale_entanglement: "whale entanglement",
  domoic_acid: "domoic acid",
  psp: "paralytic shellfish toxin (PSP)",
  regulation: "regulation",
  season: "season dates",
  quality: "quality",
  other: "other",
};
const LEVEL_SEVERITY: Record<StatusLevel, Severity> = { closed: "red", delayed: "red", restricted: "amber", watch: "amber", no_restriction_on_record: "neutral" };
const LEVEL_RANK: Record<StatusLevel, number> = { closed: 0, delayed: 0, restricted: 1, watch: 2, no_restriction_on_record: 3 };
const SEVERITY_RANK: Record<Severity, number> = { red: 0, amber: 1, neutral: 2, info: 3 };
const BASIS_RANK: Record<PortFact["basis"], number> = { official: 0, measured: 1, analysis: 2, forecast: 2, outlook: 3, derived: 4 };

/** Status older than this is flagged (curated records are re-read at least weekly). */
export const STATUS_STALE_DAYS = 14;
/** A toxin test older than this is flagged (CDPH samples sites every 1–4 weeks). */
export const TEST_STALE_DAYS = 30;
/** Neritic's own refresh older than this is flagged. */
export const DATA_STALE_HOURS = 48;
/** C-HARM probability (in [0, 1]) at or above which the run is a signal to watch (a display threshold, not a risk class). */
export const CHARM_WATCH = 0.5;

// ---------------------------------------------------------------- small pure helpers

const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const PT_DAY = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit" });
const PT_CLOCK = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit" });
const PT_LOCAL = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });

export const pacificDay = (iso: string) => (iso.length <= 10 ? iso : PT_DAY.format(new Date(iso)));
/** "Sep 28" (with the year when it is not the clock's year). */
export function dayWords(date: string, now: string): string {
  const d = pacificDay(date);
  const [y, m, dd] = d.split("-").map(Number);
  if (!y || !m || !dd) return date;
  return `${MON[m - 1]} ${dd}${String(y) !== pacificDay(now).slice(0, 4) ? `, ${y}` : ""}`;
}
/** "Dec 15–18", "Sep 28 – Oct 3" */
function rangeWords(a: string, b: string, now: string): string {
  if (a === b) return dayWords(a, now);
  const [, ma] = a.split("-").map(Number);
  const [, mb, db] = b.split("-").map(Number);
  return ma === mb && a.slice(0, 4) === b.slice(0, 4) ? `${dayWords(a, now)}–${db}` : `${dayWords(a, now)} – ${dayWords(b, now)}`;
}
/** Whole Pacific days from `date` to the clock (0 when in the future). */
export function daysOld(date: string, now: string): number {
  const a = Date.parse(`${pacificDay(date)}T00:00:00Z`);
  const b = Date.parse(`${pacificDay(now)}T00:00:00Z`);
  return Number.isFinite(a) && Number.isFinite(b) ? Math.max(0, Math.round((b - a) / 86_400_000)) : 0;
}
export const ageWords = (days: number) => (days === 0 ? "today" : days === 1 ? "yesterday" : `${days} days ago`);
const pct = (p: number) => `${Math.round(p * 100)}%`;
const ppm = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1).replace(/\.0$/, "") : v.toFixed(1));
const mg = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
/** The fact id of an NWS alert in force ("Small Craft Advisory" → "weather:alert:small-craft-advisory"). */
export const alertFactId = (event: string) => `weather:alert:${slug(event)}`;
const sentence = (s: string) => (/[.!?]$/.test(s.trim()) ? s.trim() : `${s.trim()}.`);
const lower1 = (s: string) => (/^[A-Z][a-z]/.test(s) ? s.charAt(0).toLowerCase() + s.slice(1) : s);

/** Which fishery a species name (from CDPH or the official registry) belongs to; null when none of ours. */
export function fisheryForSpecies(species: string): FisheryId | null {
  const s = species.toLowerCase();
  if (/dungeness/.test(s)) return "dungeness_commercial";
  if (/rock crab/.test(s)) return "rock_crab";
  if (/anchov|sardine/.test(s)) return "anchovy_sardine";
  if (/squid/.test(s)) return "market_squid";
  if (/mussel|clam|oyster|scallop|bivalve/.test(s)) return "bivalve_shellfish";
  if (/lobster/.test(s)) return "spiny_lobster";
  if (/salmon/.test(s)) return "salmon_commercial";
  if (/rockfish|groundfish|lingcod|sablefish|halibut|sole/.test(s)) return "groundfish";
  return null;
}
const sameFishery = (a: FisheryId, b: FisheryId) => a === b || (a.startsWith("dungeness") && b.startsWith("dungeness"));

/** CDFW's RAMP zones (14 CCR §132.8) by latitude: 1 OR border–Cape Mendocino, 2 –Pt Arena, 3 –Pigeon Pt, 4 –Lopez Pt, 5 –Pt Conception, 6 south. */
export function rampZoneFor(lat: number): { zone: string; label: string } {
  if (lat >= 40.1667) return { zone: "1", label: "Oregon border to Cape Mendocino" };
  if (lat >= 38.9583) return { zone: "2", label: "Cape Mendocino to Point Arena" };
  if (lat >= 37.1833) return { zone: "3", label: "Point Arena to Pigeon Point" };
  if (lat >= 36.0) return { zone: "4", label: "Pigeon Point to Lopez Point" };
  if (lat >= 34.45) return { zone: "5", label: "Lopez Point to Point Conception" };
  return { zone: "6", label: "Point Conception to the Mexico border" };
}

// ---------------------------------------------------------------- views the page draws

export type StatusRow = {
  fishery: FisheryId;
  label: string;
  level: StatusLevel | "unknown";
  severity: Severity;
  word: string;
  headline: string;
  cause: FisheryStatus["cause"] | null;
  causeWord: string | null;
  area: string | null;
  next_review: string | null;
  source: { name: string; url: string | null } | null;
  checked_at: string | null;
  /** null: no curated record (nothing to verify) */
  verified: boolean | null;
  /** the record shown, then any others for this fishery here */
  records: FisheryStatus[];
  /** official notices for this fishery's species tied to this place */
  notices: NoticeInput[];
  stale: boolean;
  fact: PortFact;
};

export type TestSiteView = {
  site: BiotoxinSite;
  fishery: FisheryId | null;
  latest: BiotoxinSet | null;
  action: number | null;
  sentence: string;
  ageDays: number | null;
  stale: boolean;
  over: boolean;
  /** consecutive newest sets under the action level; shown as progress toward two when relevant */
  cleanStreak: number;
  showProgress: boolean;
  progressWords: string | null;
  fact: PortFact;
};

export type BloomDay = { lead: number; date: string | null; pda: number | null; pn: number | null; basis: "cell" | "nearby" | null };
export type BloomView = {
  charm: CharmInput | null;
  days: BloomDay[];
  charmFact: PortFact;
  outlook: OutlookInput | null;
  outlookFact: PortFact | null;
  /** the published held-out skill, quoted: "RMSE ln(mg m-3): 0.754 for this outlook, 0.85 climatology, 0.892 persistence (n = 185,000; 2019 to 2021-06)" */
  skillWords: string | null;
  outlookStale: boolean;
};

export type WhaleView = {
  zone: string;
  label: string;
  action: string | null;
  depth_limit_fathoms: number | null;
  gear_reduction_pct: number | null;
  last_assessment: string | null;
  next_assessment: string | null;
  /** the assessment window as CDFW words it ("December 15-18 (then January 13-15), subject to change"), when published */
  next_window: string | null;
  season: string | null;
  source_url: string | null;
  verified: boolean | null;
  fact: PortFact;
};

export type SeaView = {
  state: "loading" | "error" | "ok";
  lanes: LanePeriod[];
  zone: string | null;
  issued: string | null;
  marineUrl: string | null;
  marineReason: string | null;
  alerts: { event: string; headline: string; url: string }[];
  buoy: { name: string; id: string; km: number; url: string; windKt: number | null; windDir: string | null; gustKt: number | null; waveFt: number | null; periodS: number | null; waterC: number | null; at: string | null } | null;
  buoyReason: string | null;
  tides: Tide[];
  tideStation: { id: string; name: string; url: string } | null;
  tideReason: string | null;
  water: { c: number; at: string; where: string; url: string; stale: boolean } | null;
};

/** The answer in one breath: `label` (the fishery, when one leads) and `text`; `status` joins them. */
export type Lede = { label: string | null; text: string; status: string; severity: Severity; more: number; sea: string | null };

export type TodayModel = {
  today: PortToday;
  lede: Lede;
  status: StatusRow[];
  statusError: boolean;
  tests: TestSiteView[];
  testsError: boolean;
  bloom: BloomView;
  whale: WhaleView | null;
  sea: SeaView;
  /** data-age facts, for the header's freshness line */
  ages: PortFact[];
  freshness: { label: string; when: string | null; stale: boolean }[];
};

// ---------------------------------------------------------------- the build

export function buildPortToday(inputs: PortTodayInputs): PortToday {
  return buildTodayModel(inputs).today;
}

export function buildTodayModel(inp: PortTodayInputs): TodayModel {
  const { place, now } = inp;
  const pid = place.place_id;
  const selected = inp.selected?.length ? inp.selected : DEFAULT_FISHERIES;
  const ages: PortFact[] = [];
  const fact = (f: Omit<PortFact, "place_id">): PortFact => ({ place_id: pid, ...f });
  const age = (id: string, severity: Severity, headline: string, detail: string | null, as_of: string | null) =>
    ages.push(fact({ id: `age:${id}`, kind: "data_age", severity, headline, detail, basis: "derived", verified: null, source: { name: "Neritic", url: "/sources" }, as_of }));

  // ---- official status per fishery (records, then notices for the fishery's species)
  const records = (inp.fisheries?.records ?? []).filter((r) => !r.applies_to_places.length || r.applies_to_places.includes(pid));
  const notices = (inp.notices ?? []).filter((n) => n.action !== "reopening" && n.action !== "advisory_lifted");
  const noticeFishery = (n: NoticeInput) => n.species.map(fisheryForSpecies).find((f): f is FisheryId => !!f) ?? null;
  const statusError = !inp.fisheries;
  if (statusError)
    age("fisheries-status", "amber", "Fishery status could not be loaded. Check CDFW and CDPH before you go.", "Neritic's season and restriction list did not load. Unknown is not open.", null);

  const status: StatusRow[] = selected.map((fishery) => {
    const label = fisheryLabel(fishery);
    const recs = records.filter((r) => r.fishery === fishery).sort((a, b) => LEVEL_RANK[a.level] - LEVEL_RANK[b.level]);
    const nts = notices.filter((n) => {
      const f = noticeFishery(n);
      return f != null && sameFishery(f, fishery) && (fishery !== "dungeness_commercial" || n.fishery !== "recreational") && (fishery !== "dungeness_recreational" || n.fishery !== "commercial");
    });
    // an official notice that closes (quarantine, closure) or restricts this fishery's species counts as a status
    const noticeLevel = (n: NoticeInput): StatusLevel => (n.action === "fishery_closure" || n.action === "quarantine" ? "closed" : "restricted");
    const topNotice = [...nts].sort((a, b) => LEVEL_RANK[noticeLevel(a)] - LEVEL_RANK[noticeLevel(b)])[0] ?? null;
    const top = recs[0] ?? null;
    const useNotice = topNotice && (!top || LEVEL_RANK[noticeLevel(topNotice)] < LEVEL_RANK[top.level]);
    let row: Omit<StatusRow, "fact">;
    if (useNotice) {
      const level = noticeLevel(topNotice);
      row = {
        fishery,
        label,
        level,
        severity: LEVEL_SEVERITY[level],
        word: LEVEL_WORD[level],
        headline: sentence(topNotice.title),
        cause: null,
        causeWord: null,
        area: null,
        next_review: null,
        source: { name: topNotice.agency, url: topNotice.url },
        checked_at: null,
        verified: topNotice.verified,
        records: recs,
        notices: nts,
        stale: false,
      };
    } else if (top) {
      const stale = daysOld(top.checked_at, now) > STATUS_STALE_DAYS;
      row = {
        fishery,
        label,
        level: top.level,
        severity: LEVEL_SEVERITY[top.level],
        word: LEVEL_WORD[top.level],
        headline: sentence(top.headline),
        cause: top.cause,
        causeWord: top.level === "no_restriction_on_record" ? null : CAUSE_WORD[top.cause],
        area: top.area_label,
        next_review: top.next_review,
        source: { name: top.source_name, url: top.source_url },
        checked_at: top.checked_at,
        verified: top.review_state === "human_verified",
        records: recs,
        notices: nts,
        stale,
      };
    } else if (statusError) {
      row = { fishery, label, level: "unknown", severity: "info", word: LEVEL_WORD.unknown, headline: "Status could not be loaded. Check CDFW.", cause: null, causeWord: null, area: null, next_review: null, source: null, checked_at: null, verified: null, records: [], notices: nts, stale: false };
    } else {
      // the calmest state: nothing on record (never "open" or "safe")
      row = {
        fishery,
        label,
        level: "no_restriction_on_record",
        severity: "neutral",
        word: LEVEL_WORD.no_restriction_on_record,
        headline: "No closure or restriction on record here.",
        cause: null,
        causeWord: null,
        area: null,
        next_review: null,
        source: { name: "CDFW and CDPH (Neritic's list)", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations" },
        checked_at: inp.fisheries?.generated_at ?? null,
        verified: null,
        records: [],
        notices: nts,
        stale: false,
      };
    }
    const nextWords = row.next_review ? ` Next review ${dayWords(row.next_review, now)}.` : "";
    const detail =
      row.level === "no_restriction_on_record"
        ? "Neritic's list has no CDFW or CDPH closure or restriction for this fishery here. That is not a guarantee: notices can lag. Check the agencies before you go."
        : `${row.records[0] && !useNotice ? sentence(row.records[0].detail) : topNotice ? sentence(topNotice.summary) : ""}${nextWords}`.trim() || null;
    return {
      ...row,
      fact: fact({
        id: `status:${fishery}`,
        kind: "status",
        severity: row.severity,
        headline: `${label}: ${row.headline}`,
        value: row.word,
        detail,
        basis: "official",
        verified: row.verified,
        source: row.source ?? { name: "CDFW", url: "https://wildlife.ca.gov" },
        as_of: row.checked_at,
      }),
    };
  });
  const staleRows = status.filter((r) => r.stale);
  if (staleRows.length) {
    const oldest = staleRows.map((r) => r.checked_at!).sort()[0];
    age("fisheries-status", "amber", `Fishery status last checked ${dayWords(oldest, now)} (${ageWords(daysOld(oldest, now))}).`, "A person transcribes these records from CDFW and CDPH; a later change may be missing. Check the agencies.", oldest);
  }

  // notices tied here (each its own fact, for alerts and Ask)
  const noticeFacts = notices.map((n) => {
    const level: StatusLevel = n.action === "fishery_closure" || n.action === "quarantine" ? "closed" : "restricted";
    return fact({ id: `notice:${n.id}`, kind: "notice", severity: LEVEL_SEVERITY[level], headline: sentence(`${n.agency}: ${n.title}`), value: LEVEL_WORD[level], detail: n.summary, basis: "official", verified: n.verified, source: { name: n.agency, url: n.url }, as_of: n.effective });
  });
  if (inp.notices === null) age("notices", "amber", "Official notices could not be loaded. Check CDFW and CDPH.", null, null);

  // ---- whale rules (RAMP)
  const whale = whaleView(inp.fisheries?.ramp ?? null, place, now, fact);

  // ---- toxin tests (CDPH)
  const tests = testViews(inp.biotoxin, place, selected, status, now, fact);
  const testsError = !inp.biotoxin;
  if (testsError) age("biotoxin", "info", "CDPH toxin test results are not available right now.", "Check CDPH's domoic acid results directly.", null);
  else if (inp.biotoxin!.source.as_of && daysOld(inp.biotoxin!.source.as_of, now) > TEST_STALE_DAYS)
    age("biotoxin", "info", `CDPH's results file is dated ${dayWords(inp.biotoxin!.source.as_of, now)}.`, "A newer CDPH file may exist.", inp.biotoxin!.source.as_of);

  // ---- bloom: C-HARM days 0–3, then the 8-day chlorophyll outlook
  const bloom = bloomView(inp.charm, inp.outlook, place, now, fact);
  if (inp.charm && (inp.charm.state === "stale" || inp.charm.state === "historical") && inp.charm.issued)
    age("charm", inp.charm.state === "historical" ? "amber" : "info", `C-HARM's newest run is from ${dayWords(inp.charm.issued, now)} (${ageWords(inp.charm.ageDays ?? daysOld(inp.charm.issued, now))}).`, inp.charm.state === "historical" ? "Too old to describe today's water." : "NOAA often skips a day; the days shown are that run's.", inp.charm.issued);

  // ---- sea and weather
  const sea = seaView(inp.conditions, now);
  const seaFacts = seaFactsOf(sea, place, now, fact);
  if (inp.conditions === "error") age("conditions", "info", "Sea and weather could not be loaded. Check NWS and NDBC.", null, null);

  if (inp.dataGeneratedAt) {
    const old = (Date.parse(now) - Date.parse(inp.dataGeneratedAt)) / 3_600_000 > DATA_STALE_HOURS;
    const when = `${dayWords(inp.dataGeneratedAt, now)}${old ? "" : `, ${PT_CLOCK.format(new Date(inp.dataGeneratedAt)).replace(/\s/g, " ")} Pacific`}`;
    age("data", "info", `Neritic's data were last refreshed ${when}.`, old ? "Newer agency information may exist." : null, inp.dataGeneratedAt);
  }

  // ---- all facts, in the strict hierarchy
  const facts = [
    ...status.map((r) => r.fact),
    ...noticeFacts,
    ...(whale ? [whale.fact] : []),
    ...tests.map((t) => t.fact),
    bloom.charmFact,
    ...(bloom.outlookFact ? [bloom.outlookFact] : []),
    ...seaFacts,
  ];
  const ordered = facts
    .map((f, i) => ({ f, i }))
    .sort((a, b) => BASIS_RANK[a.f.basis] - BASIS_RANK[b.f.basis] || SEVERITY_RANK[a.f.severity] - SEVERITY_RANK[b.f.severity] || a.i - b.i)
    .map((x) => x.f);

  const lede = ledeOf(status, statusError, sea);
  const freshness = freshnessOf(inp, sea, now);
  return {
    today: { place_id: pid, label: place.label, generated_at: now, data_generated_at: inp.dataGeneratedAt, facts: [...ordered, ...ages] },
    lede,
    status,
    statusError,
    tests,
    testsError,
    bloom,
    whale,
    sea,
    ages,
    freshness,
  };
}

// ---------------------------------------------------------------- lede

function ledeOf(rows: StatusRow[], statusError: boolean, sea: SeaView): Lede {
  const flagged = rows.filter((r) => r.severity === "red" || r.severity === "amber").sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] || LEVEL_RANK[a.level as StatusLevel] - LEVEL_RANK[b.level as StatusLevel]);
  const first = sea.lanes[0];
  const seaLine = first ? periodLine(first) : null;
  const seaWords = first && seaLine ? `${first.name}: ${seaLine}.` : null;
  if (statusError) {
    const text = "Fishery status could not be loaded. Check CDFW and CDPH before you go.";
    return { label: null, text, status: text, severity: "amber", more: 0, sea: seaWords };
  }
  if (!flagged.length) {
    const text = "No closure or restriction on record for your fisheries here. Notices can lag; check CDFW and CDPH before you go.";
    return { label: null, text, status: text, severity: "neutral", more: 0, sea: seaWords };
  }
  const top = flagged[0];
  const text = top.headline;
  return { label: top.label, text, status: `${top.label}: ${lower1(text)}`, severity: top.severity, more: flagged.length - 1, sea: seaWords };
}

// ---------------------------------------------------------------- whale (RAMP)

function whaleView(ramp: RampState | null, place: TodayPlace, now: string, fact: (f: Omit<PortFact, "place_id">) => PortFact): WhaleView | null {
  if (!ramp) return null;
  // the zone the data file maps this place to, else the zone whose published latitudes hold it, else CDFW's bounds (14 CCR §132.8)
  type Z = RampState["zones"][number] & { applies_to_places?: string[]; lat_north?: number | null; lat_south?: number | null };
  const zs = ramp.zones as Z[];
  const mapped = zs.find((x) => x.applies_to_places?.includes(place.place_id)) ?? zs.find((x) => x.lat_north != null && x.lat_south != null && place.lat <= x.lat_north && place.lat > x.lat_south) ?? null;
  const z = mapped ? { zone: mapped.zone.replace(/\D/g, "") || mapped.zone, label: mapped.label } : rampZoneFor(place.lat);
  const zone = mapped ?? zs.find((x) => x.zone.replace(/\D/g, "") === z.zone) ?? null;
  const window = (ramp as RampState & { next_assessment_window?: string | null }).next_assessment_window ?? null;
  const action = zone?.action ?? null;
  const severity: Severity = !zone ? "info" : /delay|clos/i.test(action ?? "") ? "red" : zone.depth_limit_fathoms != null || zone.gear_reduction_pct != null || /depth|gear|advis|restrict|reduc/i.test(action ?? "") ? "amber" : "neutral";
  const extras = [zone?.depth_limit_fathoms != null ? `traps limited to ${zone.depth_limit_fathoms} fathoms or shallower` : null, zone?.gear_reduction_pct != null ? `trap numbers reduced by ${zone.gear_reduction_pct}%` : null].filter(Boolean);
  const headline = zone ? `Whale rules, RAMP Zone ${z.zone} (${zone.label || z.label}): ${lower1(action ?? "no action on record").replace(/\.$/, "")}.` : `Whale rules: Neritic has no RAMP action on record for Zone ${z.zone} (${z.label}).`;
  const detail = [extras.length ? `${extras.join("; ")}.` : null, ramp.next_assessment ? `Next CDFW risk assessment ${window ?? dayWords(ramp.next_assessment, now)}.` : null, "RAMP rules apply to commercial and recreational Dungeness trap gear; they are about whale entanglement, not toxin."].filter(Boolean).join(" ");
  return {
    zone: z.zone,
    label: zone?.label || z.label,
    action,
    depth_limit_fathoms: zone?.depth_limit_fathoms ?? null,
    gear_reduction_pct: zone?.gear_reduction_pct ?? null,
    last_assessment: ramp.last_assessment,
    next_assessment: ramp.next_assessment,
    next_window: window,
    season: ramp.season,
    source_url: ramp.source_url,
    verified: ramp.review_state === "human_verified",
    fact: fact({ id: `whale:ramp:zone-${z.zone}`, kind: "whale", severity, headline, value: action, detail, basis: "official", verified: ramp.review_state === "human_verified", source: { name: "CDFW RAMP", url: ramp.source_url }, as_of: ramp.last_assessment }),
  };
}

// ---------------------------------------------------------------- toxin tests (CDPH)

function actionLevelFor(idx: BiotoxinIndex, site: BiotoxinSite, set: BiotoxinSet | null): number | null {
  if (set?.action_level_ppm != null) return set.action_level_ppm;
  const sp = site.species.toLowerCase();
  const fam = (s: string) => s.toLowerCase().replace(/s$/, "").split(/\s+/).pop() ?? s;
  const tissue = set?.tissue ?? "unknown";
  const cands = idx.action_levels.filter((a) => sp.includes(fam(a.species)) || a.species.toLowerCase().includes(fam(site.species)));
  return (cands.find((a) => a.tissue === tissue) ?? cands[0])?.ppm ?? null;
}
const setMax = (s: BiotoxinSet) => s.max_ppm ?? (s.values_ppm.filter((v): v is number => v != null).reduce((a, b) => Math.max(a, b), -Infinity) as number);
const isClean = (s: BiotoxinSet, lvl: number | null) => {
  const m = setMax(s);
  return lvl != null && Number.isFinite(m) ? m < lvl : s.values_ppm.every((v) => v == null) && s.nd_count > 0;
};

export function testSentence(site: BiotoxinSite, set: BiotoxinSet | null, lvl: number | null, now: string): string {
  // the collection site as CDPH writes it ("Monterey Bay", "Pillar Point"): more specific than the BMA
  const where = site.site_label || site.bma_label || site.port_label || "CDPH site";
  // "Dungeness", "rock crab", "northern anchovy": a proper noun keeps its capital, a common name does not
  const sp = /dungeness/i.test(site.species) ? "Dungeness" : site.species.charAt(0).toLowerCase() + site.species.slice(1);
  const subject = `${where} ${sp}`;
  if (!set) return `${subject}: no test results published.`;
  const when = dayWords(set.date, now);
  const nums = set.values_ppm.filter((v): v is number => v != null);
  if (!nums.length) return `${subject}: no domoic acid detected in ${set.nd_count || set.values_ppm.length} samples on ${when}${lvl != null ? ` (limit ${lvl} ppm)` : ""}.`;
  const max = setMax(set);
  const avg = set.avg_ppm;
  if (lvl == null) return `${subject} tested on ${when}: ${avg != null ? `avg ${ppm(avg)} ppm, ` : ""}max ${ppm(max)} ppm.`;
  if (max < lvl) return `${subject} tested ${max < lvl / 2 ? "well under" : "under"} the limit on ${when}: avg ${ppm(avg ?? max)} ppm, limit ${lvl}.`;
  const over = set.values_ppm.filter((v) => v != null && v >= lvl).length;
  return `${subject} tested over the limit on ${when}: ${over} of ${set.values_ppm.length} samples at or over ${lvl} ppm, max ${ppm(max)} ppm${avg != null ? `, avg ${ppm(avg)}` : ""}.`;
}

function testViews(idx: BiotoxinIndex | null, place: TodayPlace, selected: FisheryId[], rows: StatusRow[], now: string, fact: (f: Omit<PortFact, "place_id">) => PortFact): TestSiteView[] {
  if (!idx) return [];
  const sites = idx.sites.filter((s) => s.informs_places.includes(place.place_id));
  const views = sites.map((site): TestSiteView => {
    const sets = [...site.sets].sort((a, b) => b.date.localeCompare(a.date));
    const latest = sets[0] ?? null;
    const lvl = actionLevelFor(idx, site, latest);
    const fishery = fisheryForSpecies(site.species);
    const ageDays = latest ? daysOld(latest.date, now) : null;
    const stale = ageDays != null && ageDays > TEST_STALE_DAYS;
    const over = !!latest && lvl != null && !isClean(latest, lvl) && Number.isFinite(setMax(latest));
    let cleanStreak = 0;
    for (const s of sets) {
      if (isClean(s, actionLevelFor(idx, site, s))) cleanStreak++;
      else break;
    }
    const daRow = fishery ? rows.find((r) => sameFishery(r.fishery, fishery) && r.cause === "domoic_acid" && (r.severity === "red" || r.severity === "amber")) : null;
    const recentOver = sets.slice(0, 4).some((s) => !isClean(s, actionLevelFor(idx, site, s)) && Number.isFinite(setMax(s)));
    const showProgress = !!latest && (recentOver || !!daRow);
    const cleanDates = sets.slice(0, Math.min(cleanStreak, 2)).map((s) => dayWords(s.date, now)).reverse();
    const progressWords = !showProgress
      ? null
      : cleanStreak >= 2
        ? `Two consecutive sets under the action level (${cleanDates.join(" and ")}). Reopening is decided by CDFW and CDPH.`
        : cleanStreak === 1
          ? `1 of 2 consecutive sets under the action level so far (${cleanDates[0]}). Fisheries closed for domoic acid usually need two clean sets to reopen; CDFW and CDPH decide.`
          : "The latest set was at or over the action level. Fisheries closed for domoic acid usually need two consecutive clean sets to reopen.";
    const s = testSentence(site, latest, lvl, now);
    // a clean test never softens an official closure or delay of the same fishery (R1)
    const blocking = fishery ? rows.find((r) => sameFishery(r.fishery, fishery) && r.severity === "red") : null;
    const detail = [
      latest ? `CDPH ${latest.kind === "retest" ? "retest" : "sample set"} of ${latest.values_ppm.length} (${latest.tissue}), collected ${dayWords(latest.date, now)}${ageDays != null ? `, ${ageWords(ageDays)}` : ""}.` : null,
      stale ? `This is the latest published test; it is over ${TEST_STALE_DAYS} days old. CDPH samples every 1–4 weeks.` : null,
      blocking ? `${blocking.label} is still ${blocking.word.toLowerCase()} (${blocking.source?.name ?? "official"}); a test result does not change that.` : null,
      "A measured test of seafood tissue at one site, not a forecast.",
    ]
      .filter(Boolean)
      .join(" ");
    return {
      site,
      fishery,
      latest,
      action: lvl,
      sentence: s,
      ageDays,
      stale,
      over,
      cleanStreak,
      showProgress,
      progressWords,
      fact: fact({ id: `test:${site.id}`, kind: "test", severity: over ? "amber" : "info", headline: s, value: latest?.avg_ppm != null ? `${ppm(latest.avg_ppm)} ppm avg` : latest ? `${ppm(setMax(latest))} ppm max` : null, detail, basis: "measured", verified: null, source: { name: idx.source.name, url: idx.source.url }, as_of: latest?.date ?? null }),
    };
  });
  // in the order of the reader's fisheries (crab first by default), then any over the limit, then the newest
  const rank = (v: TestSiteView) => {
    const i = v.fishery ? selected.findIndex((f) => sameFishery(f, v.fishery!)) : -1;
    return i < 0 ? 99 : i;
  };
  return views.sort((a, b) => rank(a) - rank(b) || Number(b.over) - Number(a.over) || (b.latest?.date ?? "").localeCompare(a.latest?.date ?? ""));
}

// ---------------------------------------------------------------- bloom (C-HARM + chlorophyll outlook)

const portValue = (p: CharmProb | null): { v: number | null; basis: "cell" | "nearby" | null } => (p?.here != null ? { v: p.here, basis: "cell" } : p?.median != null ? { v: p.median, basis: "nearby" } : { v: null, basis: null });

/** The outlook's held-out skill exactly as the data file publishes it (never hard-coded). */
export function skillWords(skill: OutlookInput["skill"]): string {
  const num = (k: string) => (typeof skill[k] === "number" && Number.isFinite(skill[k]) ? (skill[k] as number) : null);
  const base = [num("climatology") != null ? `${num("climatology")} climatology` : null, num("persistence") != null ? `${num("persistence")} persistence` : null].filter(Boolean);
  const n = num("n");
  const tp = typeof skill.test_period === "string" ? skill.test_period : null;
  const tail = [n != null ? `n = ${n.toLocaleString("en-US")}` : null, tp].filter(Boolean).join("; ");
  return `Held-out ${skill.metric}: ${skill.model} for this outlook${base.length ? `, against ${base.join(" and ")}` : ""}${tail ? ` (${tail})` : ""}. Lower is better.`;
}

function bloomView(charm: CharmInput | null, outlook: OutlookInput | null, place: TodayPlace, now: string, fact: (f: Omit<PortFact, "place_id">) => PortFact): BloomView {
  const days: BloomDay[] = (charm?.days ?? []).filter((d) => d.date).map((d) => {
    const a = portValue(d.pda);
    const b = portValue(d.pn);
    return { lead: d.lead, date: d.date, pda: a.v, pn: b.v, basis: a.basis ?? b.basis };
  });
  const withPda = days.filter((d) => d.pda != null);
  let charmFact: PortFact;
  const src = { name: "NOAA C-HARM", url: charm?.url ?? "https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/california/" };
  if (!charm || charm.state === "unavailable" || !days.length) {
    charmFact = fact({ id: "charm:pda", kind: "bloom_forecast", severity: "info", headline: "No C-HARM run is available for these days.", detail: "C-HARM is NOAA's model probability of bloom conditions in the water, not toxin in seafood.", basis: "forecast", verified: null, source: src, as_of: charm?.issued ?? null, valid_for: null });
  } else if (!withPda.length) {
    charmFact = fact({ id: "charm:pda", kind: "bloom_forecast", severity: "info", headline: `C-HARM has no value within 15 km of ${place.label}.`, detail: "C-HARM's 3 km cells often stop short of the shore and of harbours. No value is not a low value.", basis: "forecast", verified: null, source: src, as_of: charm.issued, valid_for: null });
  } else {
    const vals = withPda.map((d) => d.pda!);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const range = Math.round(lo * 100) === Math.round(hi * 100) ? pct(hi) : `${pct(lo)}–${pct(hi)}`;
    const from = withPda[0].date!;
    const to = withPda[withPda.length - 1].date!;
    const historical = charm.state === "historical";
    const nearby = withPda.some((d) => d.basis === "nearby");
    charmFact = fact({
      id: "charm:pda",
      kind: "bloom_forecast",
      severity: historical ? "info" : hi >= CHARM_WATCH ? "amber" : "info",
      headline: historical
        ? `C-HARM's newest run (${charm.issued ? dayWords(charm.issued, now) : "date unknown"}) is out of date; it does not describe today's water.`
        : `C-HARM: ${range} probability of elevated particulate domoic acid in the water near ${place.label}, ${rangeWords(from, to, now)}.`,
      value: range,
      detail: `A NOAA model probability of conditions in the water (particulate domoic acid above 500 ng/L), not toxin in seafood.${nearby ? " Harbours sit inside C-HARM's coastal gap, so this is the median of its cells within 15 km, not a value at the harbour." : ""}${charm.state === "stale" ? ` From the run of ${charm.issued ? dayWords(charm.issued, now) : "an earlier day"}; NOAA has not published a newer one.` : ""}`,
      basis: "forecast",
      verified: null,
      source: src,
      as_of: charm.issued,
      valid_for: { from, to },
    });
  }
  let outlookFact: PortFact | null = null;
  let sw: string | null = null;
  let outlookStale = false;
  if (outlook) {
    sw = skillWords(outlook.skill);
    outlookStale = outlook.valid_to < pacificDay(now);
    const win = rangeWords(outlook.valid_from, outlook.valid_to, now);
    outlookFact = fact({
      id: "outlook:chl8d",
      kind: "bloom_outlook",
      severity: "info",
      headline:
        outlook.value != null
          ? `Chlorophyll outlook, ${win}: about ${mg(outlook.value)} mg/m³ near ${place.label}. A bloom indicator, not a toxin forecast.`
          : `Chlorophyll outlook, ${win}: no value near ${place.label}. A bloom indicator, not a toxin forecast.`,
      value: outlook.value != null ? `${mg(outlook.value)} mg/m³` : null,
      detail: `One 8-day mean of chlorophyll (algae of every kind), not a day-by-day forecast and not toxin. ${sw}${outlookStale ? " This outlook's window has passed." : ""}`,
      basis: "outlook",
      verified: null,
      source: { name: "Neritic chlorophyll outlook (climatology + AR(1))", url: "/research" },
      as_of: outlook.issued,
      valid_for: { from: outlook.valid_from, to: outlook.valid_to },
    });
  }
  return { charm, days, charmFact, outlook, outlookFact, skillWords: sw, outlookStale };
}

// ---------------------------------------------------------------- sea and weather

function seaView(c: Conditions | null | "error", now: string): SeaView {
  const empty: SeaView = { state: c === null ? "loading" : "error", lanes: [], zone: null, issued: null, marineUrl: null, marineReason: null, alerts: [], buoy: null, buoyReason: null, tides: [], tideStation: null, tideReason: null, water: null };
  if (c === null || c === "error") return empty;
  const v: SeaView = { ...empty, state: "ok" };
  if (c.marine.status === "ok") {
    // a period already over is not shown (a forecast issued last evening still opens with "Tonight"):
    // yesterday's night lasts until 6 AM; anything dated earlier is gone
    const today = pacificDay(now);
    const hour = Number(localNow(now).slice(11, 13));
    v.lanes = lanePeriods(c.marine.data.periods, c.marine.data.updated, 14)
      .filter((p) => !p.date || p.date >= today || (p.night && hour < 6 && daysOld(p.date, now) === 1))
      .slice(0, 10);
    v.zone = c.marine.data.zone;
    v.issued = c.marine.data.updated;
    v.marineUrl = c.marine.url;
  } else {
    v.marineReason = c.marine.reason;
    v.marineUrl = c.marine.url ?? null;
  }
  if (c.alerts.status === "ok") v.alerts = c.alerts.data.map((a) => ({ event: a.event, headline: a.headline, url: a.url }));
  if (c.buoy.status === "ok") {
    const b = c.buoy.data;
    const times = [b.windKt, b.waveFt, b.waterC].map((o) => o?.at).filter((x): x is string => !!x).sort();
    v.buoy = {
      name: c.buoy.station?.name ?? "NDBC station",
      id: c.buoy.station?.id ?? "",
      km: c.buoy.station?.km ?? 0,
      url: c.buoy.url,
      windKt: b.windKt?.value ?? null,
      windDir: b.windDirDeg ? compass(b.windDirDeg.value) : null,
      gustKt: b.gustKt?.value ?? null,
      waveFt: b.waveFt?.value ?? null,
      periodS: b.periodS?.value ?? null,
      waterC: b.waterC?.value ?? null,
      at: times.at(-1) ?? null,
    };
  } else v.buoyReason = c.buoy.reason;
  if (c.tides.status === "ok") {
    v.tides = c.tides.data;
    v.tideStation = { id: c.tides.station?.id ?? "", name: c.tides.station?.name ?? "", url: c.tides.url };
  } else v.tideReason = c.tides.reason;
  if (c.harbor?.status === "ok") v.water = { c: c.harbor.data.waterC.value, at: c.harbor.data.waterC.at, where: c.harbor.basis === "in-harbour" ? "in the harbour" : `${c.harbor.station?.km?.toFixed(1) ?? "?"} km away`, url: c.harbor.url, stale: !!c.harbor.data.stale };
  else if (v.buoy?.waterC != null && c.buoy.status === "ok" && c.buoy.data.waterC) v.water = { c: v.buoy.waterC, at: c.buoy.data.waterC.at, where: `at buoy ${v.buoy.id}`, url: v.buoy.url, stale: (Date.parse(now) - Date.parse(c.buoy.data.waterC.at)) / 3_600_000 > 3 };
  return v;
}

/** Local "YYYY-MM-DD HH:MM" in Pacific time (the CO-OPS tide table's clock). */
const localNow = (now: string) => PT_LOCAL.format(new Date(now)).replace(",", "");
export function nextTide(tides: Tide[], now: string): Tide | null {
  const n = localNow(now);
  return [...tides].sort((a, b) => a.at.localeCompare(b.at)).find((t) => t.at > n) ?? null;
}
const tideClock = (at: string) => {
  const m = /(\d{2}):(\d{2})/.exec(at);
  if (!m) return at;
  const h = Number(m[1]);
  return `${h % 12 || 12}:${m[2]} ${h < 12 ? "AM" : "PM"}`;
};

/** NWS periods after the current one that become facts (about two days). */
const NWS_MORE_PERIODS = 4;
const addDay = (ymd: string) => new Date(Date.parse(`${ymd}T12:00:00Z`) + 86_400_000).toISOString().slice(0, 10);

function seaFactsOf(sea: SeaView, place: TodayPlace, now: string, fact: (f: Omit<PortFact, "place_id">) => PortFact): PortFact[] {
  if (sea.state !== "ok") return [];
  const out: PortFact[] = [];
  const nws = { name: "NOAA National Weather Service", url: sea.marineUrl };
  for (const a of sea.alerts) out.push(fact({ id: alertFactId(a.event), kind: "weather", severity: "amber", headline: sentence(`NWS: ${a.headline}`), detail: "In effect for these waters, as issued by NWS.", basis: "official", verified: null, source: { name: nws.name, url: a.url }, as_of: sea.issued }));
  const first = sea.lanes[0];
  if (first) {
    const line = periodLine(first);
    out.push(fact({ id: "weather:nws", kind: "weather", severity: "info", headline: `NWS ${sea.zone ?? "marine"} forecast, ${first.name}: ${line ?? first.text}`, value: line, detail: first.text, basis: "forecast", verified: null, source: nws, as_of: sea.issued, valid_for: null }));
  }
  // the periods that follow, each with the day it is for, so "tomorrow" can be answered from NWS's own words
  for (const p of sea.lanes.slice(1, 1 + NWS_MORE_PERIODS)) {
    if (!p.date) continue;
    const line = periodLine(p);
    out.push(fact({ id: `weather:nws:${p.date}:${p.night ? "night" : "day"}`, kind: "weather", severity: "info", headline: `NWS ${sea.zone ?? "marine"} forecast, ${p.name}: ${line ?? p.text}`, value: line, detail: p.text, basis: "forecast", verified: null, source: nws, as_of: sea.issued, valid_for: { from: p.date, to: p.date } }));
  }
  if (sea.buoy) {
    const b = sea.buoy;
    const parts = [b.windKt != null ? `wind ${b.windDir ? `${b.windDir} ` : ""}${Math.round(b.windKt)} kt${b.gustKt != null ? ` (gust ${Math.round(b.gustKt)})` : ""}` : null, b.waveFt != null ? `waves ${b.waveFt.toFixed(1)} ft${b.periodS != null ? ` at ${Math.round(b.periodS)} s` : ""}` : null].filter(Boolean);
    if (parts.length) out.push(fact({ id: "sea:buoy", kind: "observation", severity: "info", headline: `Measured at buoy ${b.id}: ${parts.join(", ")}.`, value: parts.join(", "), detail: `NDBC station ${b.id} (${b.name}), ${Math.round(b.km)} km from ${place.label}. A measurement at one point, not a forecast.`, basis: "measured", verified: null, source: { name: "NOAA NDBC", url: b.url }, as_of: b.at }));
  }
  const nt = nextTide(sea.tides, now);
  if (nt) out.push(fact({ id: "tide:next", kind: "tide", severity: "info", headline: `Next tide: ${nt.type === "H" ? "high" : "low"} ${nt.ft.toFixed(1)} ft at ${tideClock(nt.at)}.`, value: `${nt.ft.toFixed(1)} ft`, detail: `NOAA CO-OPS astronomical prediction at station ${sea.tideStation?.id ?? ""}; weather can shift real water levels.`, basis: "forecast", verified: null, source: { name: "NOAA CO-OPS", url: sea.tideStation?.url ?? null }, as_of: nt.at.slice(0, 10), valid_for: null }));
  // today's and tomorrow's highs and lows, each day one fact
  const today = pacificDay(now);
  for (const day of [today, addDay(today)]) {
    const list = sea.tides.filter((t) => t.at.startsWith(day)).sort((a, b) => a.at.localeCompare(b.at));
    if (!list.length) continue;
    const words = list.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft at ${tideClock(t.at)}`).join(", ");
    out.push(fact({ id: `tide:day:${day}`, kind: "tide", severity: "info", headline: `Tides ${day === today ? "today" : "tomorrow"} (${dayWords(day, now)})${sea.tideStation?.name ? ` at ${sea.tideStation.name}` : ""}: ${words}.`, value: null, detail: "NOAA CO-OPS astronomical predictions, feet above MLLW; wind and swell shift real water levels.", basis: "forecast", verified: null, source: { name: "NOAA CO-OPS", url: sea.tideStation?.url ?? null }, as_of: day, valid_for: { from: day, to: day } }));
  }
  if (sea.water) {
    const f = (sea.water.c * 9) / 5 + 32;
    out.push(fact({ id: "sea:water", kind: "observation", severity: "info", headline: `Water temperature ${Math.round(f)} °F (${sea.water.c.toFixed(1)} °C), measured ${sea.water.where}.`, value: `${Math.round(f)} °F`, detail: sea.water.stale ? "The latest reading is more than an hour old." : null, basis: "measured", verified: null, source: { name: "NOAA", url: sea.water.url }, as_of: sea.water.at }));
  }
  return out;
}

// ---------------------------------------------------------------- freshness, in plain words

function freshnessOf(inp: PortTodayInputs, sea: SeaView, now: string): { label: string; when: string | null; stale: boolean }[] {
  const out: { label: string; when: string | null; stale: boolean }[] = [];
  const checked = (inp.fisheries?.records ?? []).map((r) => r.checked_at).sort().at(-1) ?? null;
  out.push({ label: "Status", when: checked ? `checked ${dayWords(checked, now)}` : inp.fisheries ? "no records" : "not available", stale: !inp.fisheries || (!!checked && daysOld(checked, now) > STATUS_STALE_DAYS) });
  if (sea.issued) out.push({ label: "NWS", when: PT_CLOCK.format(new Date(sea.issued)).replace(/\s/g, " ") + (pacificDay(sea.issued) === pacificDay(now) ? "" : ` ${dayWords(sea.issued, now)}`), stale: (Date.parse(now) - Date.parse(sea.issued)) / 3_600_000 > 12 });
  if (sea.buoy?.at) {
    const m = Math.max(0, Math.round((Date.parse(now) - Date.parse(sea.buoy.at)) / 60000));
    out.push({ label: "Buoy", when: m < 60 ? `${m} min ago` : `${Math.floor(m / 60)} h ago`, stale: m > 180 });
  }
  const newestTest = (inp.biotoxin?.sites ?? []).filter((s) => s.informs_places.includes(inp.place.place_id)).flatMap((s) => s.sets.map((x) => x.date)).sort().at(-1);
  if (newestTest) out.push({ label: "Tests", when: dayWords(newestTest, now), stale: daysOld(newestTest, now) > TEST_STALE_DAYS });
  if (inp.charm?.issued) out.push({ label: "C-HARM", when: `run ${dayWords(inp.charm.issued, now)}`, stale: inp.charm.state !== "current" });
  return out;
}

// ---------------------------------------------------------------- what changed since the last visit

export type FactSnap = { id: string; kind: FactKind; headline: string; value: string | null; severity: Severity; as_of: string | null };
export type FactChange = { id: string; change: "new" | "changed" | "gone"; before: FactSnap | null; after: FactSnap | null };

/** Kinds that count as news between visits (the sea, tides and buoy change hourly and are not listed). */
const CHANGE_KINDS: FactKind[] = ["status", "notice", "whale", "test", "bloom_forecast", "bloom_outlook"];
const counts = (f: { id: string; kind: FactKind }) => CHANGE_KINDS.includes(f.kind) || f.id.startsWith("weather:alert:");

export function snapFacts(facts: PortFact[]): FactSnap[] {
  return facts.filter(counts).map((f) => ({ id: f.id, kind: f.kind, headline: f.headline, value: f.value ?? null, severity: f.severity, as_of: f.as_of }));
}

/**
 * The differences between the facts a reader saw last time (kept on their device) and now, keyed by
 * PortFact.id. A model run or outlook counts as changed only when a newer one was issued (its `as_of`);
 * everything else when its words, value or severity changed.
 */
export function diffFacts(prev: FactSnap[], next: PortFact[]): FactChange[] {
  const before = new Map(prev.map((f) => [f.id, f]));
  const now = snapFacts(next);
  const after = new Map(now.map((f) => [f.id, f]));
  const out: FactChange[] = [];
  for (const f of now) {
    const p = before.get(f.id);
    if (!p) out.push({ id: f.id, change: "new", before: null, after: f });
    else {
      const modelled = f.kind === "bloom_forecast" || f.kind === "bloom_outlook";
      const changed = modelled ? p.as_of !== f.as_of : p.headline !== f.headline || p.value !== f.value || p.severity !== f.severity;
      if (changed) out.push({ id: f.id, change: "changed", before: p, after: f });
    }
  }
  for (const p of prev) if (!after.has(p.id) && counts(p)) out.push({ id: p.id, change: "gone", before: p, after: null });
  const rank = (c: FactChange) => SEVERITY_RANK[(c.after ?? c.before)!.severity] + (c.change === "gone" ? 0.5 : 0);
  return out.sort((a, b) => rank(a) - rank(b));
}
