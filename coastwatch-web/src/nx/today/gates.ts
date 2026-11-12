/**
 * Today's three gates (v13 lead decision 6): what decides the reader's next trip, each fact under the gate it
 * informs, so no two unrelated figures share a headline. PURE: `buildToday` (model.ts) hands it words and
 * numbers it already took from the record; this file only arranges them and adds the FIXED sentences that say
 * how the gates relate.
 *
 *   whale    CDFW's RAMP measures for the zone (Dungeness readers only): the delay's own cause, the next
 *            assessment date
 *   domoic   CDPH's latest test for the fishery: the highest result, its date and age. A test older than
 *            TEST_STALE_DAYS is history: it is worded as old and is never a figure on the chart
 *   sea      NWS advisories FIRST, then today's forecast, the buoy's waves and wind, the water, the tide
 *
 * Relationship sentences are fixed and never computed from the values (T41, T43, T44): a test, at any level,
 * does not lift a whale delay; an old test is not the water now; the sea and the season are separate.
 */
import type { Provenance } from "@/nx/ds/Stamp";
import { alertsHref } from "@/nx/shell/unread";

export type GateId = "whale" | "domoic" | "sea";
export type GateState = "ok" | "old" | "loading" | "unavailable" | "none";

export type GateAction = { id: string; label: string; href: string; external?: boolean };

export type GateFact = {
  id: string;
  /** what kind of line it is: an advisory leads the sea gate */
  role: "advisory" | "forecast" | "measured" | "predicted" | "record" | "note";
  text: string;
  /** past its stale limit: said as "(old)" beside the words (T17) */
  old?: boolean;
  provenance?: Provenance | null;
};

export type Gate = {
  id: GateId;
  /** "Whale measures", "Domoic acid", "Sea" */
  title: string;
  /** who decides or measures: "CDFW RAMP", "CDPH tests", "NWS and NDBC" */
  authority: string;
  state: GateState;
  /** the gate in one line (for the sea gate, the words of its first fact: an advisory when one is in force) */
  headline: string;
  /** the date the headline rests on (ISO), what it is, and its age in words ("45 days ago") */
  date: string | null;
  dateRole: Provenance["dateRole"] | null;
  age: string | null;
  facts: GateFact[];
  /** how this gate relates to the others: fixed sentences */
  relation: string[];
  actions: GateAction[];
  provenance: Provenance | null;
};

export const RELATION = {
  /** the season status on this page is a whale decision */
  whaleDecides: "This is what delays the season: a whale-entanglement decision by CDFW, not a toxin result.",
  whaleGeneral: "RAMP is CDFW's whale-entanglement programme. It is separate from domoic acid testing.",
  /** T44: a test never softens a whale delay or closure */
  testDoesNotLift: "A test result, at any level, does not lift the whale delay: that is a separate decision by CDFW.",
  testOld: (days: number) => `This test is older than ${days} days: it is history, not a reading of the water now.`,
  /** T43: "usually", and the agencies decide */
  testWhoDecides: "CDPH tests and CDFW acts on the results; they usually look for two consecutive sets under the action level.",
  seaSeparate: "The sea and the season are separate: neither changes the other.",
} as const;

const q = (key: string) => encodeURIComponent(key);

/** Where each gate's next steps lead; every in-app address keeps the port (`place`). */
export const gateHrefs = {
  // the shell's own address for a port's alerts (the bar's link uses the same one)
  alerts: (place: string) => alertsHref(place),
  /** water samples near the port, opened on particulate domoic acid (the Explore → Research contract's `var`) */
  research: (place: string) => `/research?place=${q(place)}&var=pDA`,
  explore: (place: string) => `/explore?place=${q(place)}&layer=sst`,
  trip: (place: string) => `/coast?place=${q(place)}&view=trip`,
  notices: (place: string) => `/notices?place=${q(place)}`,
};

export type WhaleInput = {
  zone: string;
  /** "Fleet Advisory, no depth or gear limit on record" */
  short: string;
  /** the season status on this page is itself a whale delay or closure */
  decidesSeason: boolean;
  assessed: string | null;
  age: string | null;
  sourceUrl: string | null;
  provenance: Provenance | null;
  /** "Next whale assessment, subject to change: Dec 15–18" */
  next: string | null;
  /** CDFW's own clauses for the reader's sector, quoted, and the zone's extent */
  record: GateFact[];
};

export type TestInput =
  | { state: "error" }
  | { state: "none" }
  | {
      state: "ok";
      /** "Dungeness test, Monterey Bay: highest 9.8 of 30 ppm, 6 samples" */
      headline: string;
      date: string;
      age: string;
      old: boolean;
      staleDays: number;
      progress: string | null;
      sourceUrl: string | null;
      provenance: Provenance | null;
      /** the port's other CDPH test sites, each in one sentence with its age */
      others: GateFact[];
    };

export type SeaInput =
  | { state: "loading" }
  | { state: "error" }
  | { state: "ok"; facts: GateFact[]; marineUrl: string | null; issued: string | null; age: string | null };

export type GateInput = {
  place: string;
  whale: WhaleInput | null;
  test: TestInput;
  sea: SeaInput;
};

const ROLE_ORDER: GateFact["role"][] = ["advisory", "forecast", "measured", "predicted", "record", "note"];

/** Advisories first, then the forecast, then what was measured; the order within a role is kept. */
export function orderSeaFacts(facts: GateFact[]): GateFact[] {
  return facts.map((f, i) => ({ f, i })).sort((a, b) => ROLE_ORDER.indexOf(a.f.role) - ROLE_ORDER.indexOf(b.f.role) || a.i - b.i).map((x) => x.f);
}

function whaleGate(w: WhaleInput, place: string): Gate {
  const actions: GateAction[] = [];
  if (w.sourceUrl && /^https?:/.test(w.sourceUrl)) actions.push({ id: "notice", label: "Read CDFW's whale measures", href: w.sourceUrl, external: true });
  actions.push({ id: "alerts", label: "Alerts for this port", href: gateHrefs.alerts(place) });
  return {
    id: "whale",
    title: "Whale measures",
    authority: "CDFW RAMP",
    state: "ok",
    headline: `Zone ${w.zone}: ${w.short}`,
    date: w.assessed,
    dateRole: w.assessed ? "assessed" : null,
    age: w.age,
    facts: [...(w.next ? [{ id: "next", role: "record" as const, text: w.next, provenance: w.provenance }] : []), ...w.record],
    relation: [w.decidesSeason ? RELATION.whaleDecides : RELATION.whaleGeneral],
    actions,
    provenance: w.provenance,
  };
}

function domoicGate(t: TestInput, whaleDelay: boolean, place: string): Gate {
  const research: GateAction = { id: "research", label: "Water samples nearby in Research", href: gateHrefs.research(place) };
  const base = { id: "domoic" as const, title: "Domoic acid", authority: "CDPH tests" };
  if (t.state === "error")
    return { ...base, state: "unavailable", headline: "CDPH test results could not be read just now.", date: null, dateRole: null, age: null, facts: [], relation: whaleDelay ? [RELATION.testDoesNotLift] : [], actions: [research], provenance: null };
  if (t.state === "none")
    return { ...base, state: "none", headline: "No CDPH test site is tied to this port.", date: null, dateRole: null, age: null, facts: [], relation: [RELATION.testWhoDecides, ...(whaleDelay ? [RELATION.testDoesNotLift] : [])], actions: [research], provenance: null };
  const relation = [...(t.old ? [RELATION.testOld(t.staleDays)] : []), ...(whaleDelay ? [RELATION.testDoesNotLift] : []), RELATION.testWhoDecides];
  const actions: GateAction[] = [];
  if (t.sourceUrl && /^https?:/.test(t.sourceUrl)) actions.push({ id: "cdph", label: "CDPH's results", href: t.sourceUrl, external: true });
  actions.push(research);
  return {
    ...base,
    state: t.old ? "old" : "ok",
    headline: t.headline,
    date: t.date,
    dateRole: "sampled",
    age: t.age,
    facts: [...(t.progress ? [{ id: "progress", role: "record" as const, text: t.progress, old: t.old, provenance: t.provenance }] : []), ...t.others],
    relation,
    actions,
    provenance: t.provenance,
  };
}

function seaGate(s: SeaInput, place: string): Gate {
  const base = { id: "sea" as const, title: "Sea", authority: "NWS and NDBC" };
  const actions: GateAction[] = [
    { id: "explore", label: "Conditions on the chart in Explore", href: gateHrefs.explore(place) },
    { id: "trip", label: "Plan a trip (hypothetical)", href: gateHrefs.trip(place) },
  ];
  if (s.state === "loading") return { ...base, state: "loading", headline: "Reading NWS, the buoy and the tide", date: null, dateRole: null, age: null, facts: [], relation: [RELATION.seaSeparate], actions, provenance: null };
  if (s.state === "error")
    return { ...base, state: "unavailable", headline: "NWS, the buoy and the tide could not be read just now.", date: null, dateRole: null, age: null, facts: [], relation: [RELATION.seaSeparate], actions: [{ id: "nws", label: "NWS marine forecast", href: "https://www.weather.gov/marine", external: true }, ...actions], provenance: null };
  const facts = orderSeaFacts(s.facts);
  const lead = facts[0];
  if (s.marineUrl && /^https?:/.test(s.marineUrl)) actions.unshift({ id: "nws", label: "NWS marine forecast", href: s.marineUrl, external: true });
  return {
    ...base,
    state: facts.length ? "ok" : "none",
    headline: lead ? lead.text : "No NWS forecast, buoy or tide for this port just now.",
    date: s.issued,
    dateRole: s.issued ? "issued" : null,
    age: s.age,
    facts,
    relation: [RELATION.seaSeparate],
    actions,
    provenance: lead?.provenance ?? null,
  };
}

/** The gates in their fixed order: whale (when RAMP applies), domoic acid, sea. */
export function buildGates(inp: GateInput): Gate[] {
  const whaleDelay = !!inp.whale?.decidesSeason;
  return [...(inp.whale ? [whaleGate(inp.whale, inp.place)] : []), domoicGate(inp.test, whaleDelay, inp.place), seaGate(inp.sea, inp.place)];
}

/** NWS advisories in force, for the top of the page (they are also the sea gate's first lines). */
export function advisoriesOf(gates: Gate[]): GateFact[] {
  const sea = gates.find((g) => g.id === "sea");
  if (!sea || sea.state !== "ok") return [];
  return sea.facts.filter((f) => f.role === "advisory");
}
