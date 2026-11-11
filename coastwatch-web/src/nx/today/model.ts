/**
 * Today at a port, for one fishery: the v12 view model (docs/coastwatch/v12/design-system/journeys.md,
 * Journeys 1 and 2). PURE: it reads the pilot's one fact model (src/pilot/facts.ts `buildTodayModel`, the
 * same facts alerts and Ask use) and arranges it for the chart and the margin. It never adds a fact of its
 * own: every sentence below is either the record's words, a number from the record, or a fixed label.
 *
 *   statement   the season status for the reader's fishery, its why, its stamp (official, not verified)
 *   decision    the next date that can change it (BigDate), only when the record names one in the future
 *   agenda      dated items: the last notice, the latest test (with age), today's NWS and tides, coming
 *               NWS periods that matter, later assessments
 *   soundings   at most two figures at their places: the test area, the buoy (README principle 2)
 *   rows        everything else, one line each, expanding in place (whale, tests, water, models, others)
 *
 * Hierarchy (T44): official status, then measured tests and observations, then models. A test never
 * softens the status; a model never alarms (T12).
 */
import type { FisheryId, PortFact, StatusLevel } from "@/pilot/contracts";
import type { Conditions } from "@/lib/conditions";
import { buildTodayModel, FISHERY_OPTIONS, fisheryLabel, nextTide, pacificDay, rowHeadline, type PortTodayInputs, type StatusRow, type TestSiteView, type TodayModel } from "@/pilot/facts";
import { periodLine, type LanePeriod } from "@/pilot/today/marine";
import type { Provenance } from "@/nx/ds/Stamp";
import { age as ageWords, day } from "@/nx/ds/format";
import { provenanceOf, shortSource } from "./provenance";

export type TodayInputs = Omit<PortTodayInputs, "conditions" | "selected">;
export type SeaState = Conditions | null | "error";

export type Tone = "ink" | "regulation" | "closure";

export type StatementView =
  | {
      state: "ok";
      fishery: FisheryId;
      fisheryLabel: string;
      level: StatusLevel;
      title: string;
      why: string | null;
      tone: Tone;
      provenance: Provenance | null;
      /** the agency page the record quotes ("Read CDFW's notice") */
      source: { agency: string; url: string } | null;
      conflict: StatusRow["conflict"];
      row: StatusRow;
    }
  | { state: "unavailable"; fishery: FisheryId; fisheryLabel: string };

export type DecisionView = {
  label: string;
  start: string;
  end: string | null;
  tentative: boolean;
  provenance: Provenance | null;
  /** the status record names the same date from the same agency: one stamp, after the date, covers both */
  sameAs?: boolean;
};

export type AgendaDatum = {
  id: string;
  start: string;
  end?: string | null;
  title: string;
  detail?: string | null;
  tone?: Tone;
  tentative?: boolean;
  provenance?: Provenance | null;
};

export type SoundingDatum = {
  id: "test" | "buoy";
  lon: number;
  lat: number;
  label: string;
  value: string;
  unit: string | null;
  caption: string[];
  /** phones: the figure with one short line under it (kind and source, date); the rest is in the provenance */
  short: string;
  stale: boolean;
  /** the full spoken reading */
  spoken: string;
  provenance: Provenance | null;
};

export type RowDatum =
  | { id: string; label: string; state: "ok" | "stale"; value: string; detail: Array<{ text: string; provenance?: Provenance | null }>; provenance?: Provenance | null }
  | { id: string; label: string; state: "unavailable"; note: string };

export type OtherFishery = { fishery: FisheryId; label: string; word: string; headline: string; flagged: boolean; provenance: Provenance | null };

export type ModelsView = {
  /** the row's one-line value */
  value: string;
  charm: { headline: string; words: string; past: boolean; provenance: Provenance | null } | null;
  outlook: { headline: string; words: string[]; provenance: Provenance | null; signal: "higher" | "lower" | "none" | "unknown" } | null;
};

export type TodayView = {
  place: { key: string; name: string; lat: number; lon: number };
  now: string;
  fishery: FisheryId;
  statement: StatementView;
  decision: DecisionView | null;
  agenda: AgendaDatum[];
  sea: "loading" | "error" | "ok";
  soundings: SoundingDatum[];
  rows: RowDatum[];
  models: ModelsView;
  others: OtherFishery[];
  /** every fishery's facts (for the change line: a change of fishery is not news) */
  allFacts: PortFact[];
  dataAt: string | null;
  model: TodayModel;
};

// ---------------------------------------------------------------- small words

const LEVEL_NOUN: Record<StatusLevel, string> = { delayed: "Delay", closed: "Closure", restricted: "Restriction", watch: "Advisory", no_restriction_on_record: "Record" };
const cap1 = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const strip = (s: string) => s.trim().replace(/[.]+$/, "");
const MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
const ymd = (y: number, m: number, d: number) => `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;

/** "Dungeness crab (commercial)" → "Dungeness crab, commercial": the scope line's words. */
export const scopeWords = (label: string) => label.replace(/\s*\((\w+)\)$/, ", $1");

/**
 * A headline split into the statement and its reason: "Commercial season delayed to at least Jan 1, 2027
 * (humpback whale entanglements)." → title "Commercial season delayed to at least Jan 1, 2027", why
 * "Humpback whale entanglements". Only a trailing parenthetical is split; the words are the record's.
 */
export function splitHeadline(h: string): { title: string; why: string | null } {
  const m = strip(h).match(/^(.*\S)\s*\(([^()]{3,80})\)$/);
  if (!m) return { title: strip(h), why: null };
  return { title: m[1], why: cap1(m[2]) };
}

/**
 * CDFW's assessment window as dates: "December 15-18 (then January 13-15), subject to change", anchored on
 * the record's own `next_assessment` (its year, and its month for what follows). Unreadable text gives [].
 */
export function parseWindow(text: string | null | undefined, anchor: string | null | undefined): Array<{ start: string; end: string | null }> {
  if (!text || !anchor || !/^\d{4}-\d{2}-\d{2}/.test(anchor)) return [];
  const y0 = Number(anchor.slice(0, 4));
  const m0 = Number(anchor.slice(5, 7)) - 1;
  const out: Array<{ start: string; end: string | null }> = [];
  const re = /\b([A-Z][a-z]+)\.?\s+(\d{1,2})(?:\s*[-–]\s*(?:([A-Z][a-z]+)\.?\s+)?(\d{1,2}))?/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    const a = MONTHS.findIndex((x) => x.startsWith(m![1].toLowerCase().slice(0, 3)));
    if (a < 0) continue;
    const b = m[3] ? MONTHS.findIndex((x) => x.startsWith(m![3].toLowerCase().slice(0, 3))) : a;
    if (b < 0) continue;
    const ya = a < m0 ? y0 + 1 : y0;
    const yb = b < a ? ya + 1 : ya;
    out.push({ start: ymd(ya, a, Number(m[2])), end: m[4] ? ymd(yb, b, Number(m[4])) : null });
  }
  return out;
}

const tideClock = (at: string) => {
  const m = /(\d{2}):(\d{2})/.exec(at);
  if (!m) return at;
  const h = Number(m[1]);
  return `${h % 12 || 12}:${m[2]} ${h < 12 ? "AM" : "PM"}`;
};
/** "high 5.8 ft at 9:54 AM": a CO-OPS tide turn in the station's own clock (Pacific local). */
const tideWords = (t: { at: string; ft: number; type: "H" | "L" }) => `${t.type === "H" ? "high" : "low"} ${t.ft < 0 ? "−" : ""}${Math.abs(t.ft).toFixed(1)} ft at ${tideClock(t.at)}`;

// ---------------------------------------------------------------- the status

function toneOf(r: StatusRow): Tone {
  if (r.conflict) return "ink";
  if (r.level === "closed" && !r.seasonal) return "closure";
  if (r.level === "delayed" || r.level === "restricted") return "regulation";
  return "ink";
}

function statementOf(row: StatusRow | undefined, fishery: FisheryId, statusError: boolean): StatementView {
  const label = fisheryLabel(fishery);
  if (!row || row.level === "unknown" || (statusError && !row.source)) return { state: "unavailable", fishery, fisheryLabel: label };
  let title: string;
  let why: string | null;
  if (row.conflict) {
    title = "Sources differ";
    why = "Awaiting a person's review. Neither record is chosen here.";
  } else if (row.level === "no_restriction_on_record") {
    title = "No closure or restriction on record";
    why = row.records.length ? strip(row.headline) : `Neritic's list has nothing for ${label.toLowerCase().replace(/\s*\(.*\)$/, "")} here. That does not mean the season is open.`;
  } else {
    ({ title, why } = splitHeadline(rowHeadline(row)));
    // the scope line already names the sector: "Commercial season delayed…" reads "Season delayed…"
    const sector = label.match(/\((\w+)\)$/)?.[1];
    if (sector && title.toLowerCase().startsWith(`${sector.toLowerCase()} `)) title = cap1(title.slice(sector.length + 1));
    if (row.seasonal && !why) why = "By the season's own dates";
  }
  const p = provenanceOf(row.fact, label, { value: row.word, sourceUrl: row.source?.url ?? undefined, place: row.area ?? undefined });
  return {
    state: "ok",
    fishery,
    fisheryLabel: label,
    level: row.level as StatusLevel,
    title,
    why,
    tone: toneOf(row),
    provenance: p,
    source: row.source?.url && /^https?:/.test(row.source.url) ? { agency: shortSource(row.source.name), url: row.source.url } : null,
    conflict: row.conflict,
    row,
  };
}

const isDungeness = (f: FisheryId) => f === "dungeness_commercial" || f === "dungeness_recreational";

/** The next date that can change this fishery's status, as the record names it; never a past one. */
function decisionOf(model: TodayModel, row: StatusRow | undefined, fishery: FisheryId, inp: TodayInputs): { decision: DecisionView | null; later: AgendaDatum[] } {
  const today = pacificDay(inp.now);
  const ramp = inp.fisheries?.ramp as { checked_at?: string } | null | undefined;
  const w = model.whale;
  if (isDungeness(fishery) && w?.next_assessment && w.next_assessment >= today) {
    const spans = parseWindow(w.next_window, w.next_assessment);
    const tentative = /subject to change/i.test(w.next_window ?? "") || !w.next_window;
    const first = spans[0] ?? { start: w.next_assessment, end: null };
    const checked = ramp?.checked_at ?? w.last_assessment;
    const prov: Provenance | null = checked
      ? {
          kind: "official",
          title: `Whale assessment, RAMP zone ${w.zone}`,
          value: w.next_window ?? undefined,
          source: "CDFW RAMP",
          sourceLong: "CDFW Risk Assessment Mitigation Program (whale entanglement)",
          sourceUrl: w.source_url ?? undefined,
          date: checked,
          dateRole: "checked",
          place: `Zone ${w.zone}: ${w.label}`,
          caveats: ["RAMP is about whale entanglement, not domoic acid.", ...(tentative ? ["CDFW says these dates are subject to change."] : [])],
        }
      : null;
    return {
      decision: { label: `Next whale assessment${tentative ? ", subject to change" : ""}`, start: first.start, end: first.end, tentative, provenance: prov, sameAs: !!row && row.cause === "whale_entanglement" && !!row.next_review && pacificDay(row.next_review) === first.start && !!row.source && /CDFW/.test(row.source.name) },
      later: spans.slice(1).map((s, i) => ({ id: `assessment-${i + 1}`, start: s.start, end: s.end, title: "Following whale assessment", tentative, tone: "regulation" as Tone, provenance: prov })),
    };
  }
  if (row?.next_review && pacificDay(row.next_review) >= today) {
    return { decision: { label: `Next review by ${row.source ? shortSource(row.source.name) : "the agency"}`, start: pacificDay(row.next_review), end: null, tentative: false, provenance: provenanceOf(row.fact, row.label) }, later: [] };
  }
  return { decision: null, later: [] };
}

// ---------------------------------------------------------------- tests

/** The test that matters to this fishery: its own species first, else the first the port has (crab first). */
function testFor(model: TodayModel, fishery: FisheryId): TestSiteView | null {
  const own = model.tests.find((t) => t.fishery && (t.fishery === fishery || (isDungeness(t.fishery) && isDungeness(fishery))));
  return own ?? model.tests[0] ?? null;
}

const speciesShort = (s: string) => (/dungeness/i.test(s) ? "Dungeness" : cap1(s.toLowerCase()));
const ppmWords = (v: number) => String(Number(v.toPrecision(3)));
const nums = (t: TestSiteView) => (t.latest?.values_ppm ?? []).filter((v): v is number => v != null);
const maxOf = (t: TestSiteView) => t.latest?.max_ppm ?? (nums(t).length ? Math.max(...nums(t)) : null);

/** "All 6 under the action level", "2 of 6 at or over the action level", "None detected": no figure (the sounding holds it). */
function testPhrase(t: TestSiteView): string {
  const set = t.latest!;
  const n = set.values_ppm.length;
  const vals = nums(t);
  if (!vals.length) return set.nd_count ? `None detected in ${set.nd_count || n}` : "No value published";
  if (t.action == null) return `${n} sample${n === 1 ? "" : "s"}`;
  const over = vals.filter((v) => v >= t.action!).length;
  return over ? `${over} of ${n} at or over the action level` : `All ${n} under the action level`;
}

function testProvenance(t: TestSiteView): Provenance | null {
  if (!t.latest) return null;
  return provenanceOf(t.fact, `${t.site.site_label} ${speciesShort(t.site.species).toLowerCase() === "dungeness" ? "Dungeness" : t.site.species.toLowerCase()}, CDPH test`, {
    value: t.sentence,
    place: [t.site.site_label, t.site.bma_label && t.site.bma_label !== t.site.site_label ? `BMA ${t.site.bma ?? ""} ${t.site.bma_label}`.trim() : null].filter(Boolean).join(", "),
    method: `${t.latest.values_ppm.length} samples of ${t.latest.tissue}, CDPH laboratory${t.action != null ? `; action level ${t.action} ppm` : ""}`,
    stale: t.stale,
  });
}

// ---------------------------------------------------------------- the build

export type BuildOptions = { fishery: FisheryId; sea: SeaState; place: { key: string; name: string } };

export function buildToday(inp: TodayInputs, o: BuildOptions): TodayView {
  const now = inp.now;
  const today = pacificDay(now);
  const others = FISHERY_OPTIONS.map((f) => f.id).filter((f) => f !== o.fishery);
  const model = buildTodayModel({ ...inp, conditions: o.sea, selected: [o.fishery, ...others] });
  const row = model.status.find((r) => r.fishery === o.fishery);
  const statement = statementOf(row, o.fishery, model.statusError);
  const { decision, later } = decisionOf(model, row, o.fishery, inp);

  // ---- agenda
  const agenda: AgendaDatum[] = [];
  if (row && statement.state === "ok" && row.level !== "no_restriction_on_record") {
    const rec = row.records[0];
    const notice = row.notices[0];
    // the record the row shows (a curated record carries a cause; an official notice does not)
    const eff = row.cause !== null && rec ? rec.effective : (notice?.effective ?? rec?.effective ?? null);
    if (eff) {
      const past = pacificDay(eff) <= today;
      const agency = row.source ? shortSource(row.source.name) : "the agency";
      agenda.push({ id: "notice", start: pacificDay(eff), title: `${LEVEL_NOUN[row.level as StatusLevel]} ${past ? "took effect" : "takes effect"}, ${agency}`, tone: statement.tone === "ink" ? "ink" : statement.tone, provenance: statement.provenance });
    }
  }
  const test = testFor(model, o.fishery);
  const testHasPlace = !!test?.latest && test.site.lat != null && test.site.lon != null;
  if (test?.latest) {
    agenda.push({
      id: "test",
      start: test.latest.date,
      title: `CDPH test, ${test.site.site_label} ${speciesShort(test.site.species) === "Dungeness" ? "Dungeness" : test.site.species.toLowerCase()}`,
      detail: `${testHasPlace ? testPhrase(test) : test.sentence.replace(/^[^:]*:\s*/, "")}. ${cap1(ageWords(test.latest.date, now))}${test.stale ? " (old)" : ""}.`.replace(/\.\./g, "."),
      provenance: testProvenance(test),
    });
  }
  const sea = model.sea;
  if (sea.state === "ok") {
    const nws = model.today.facts.find((f) => f.id === "weather:nws");
    const nwsProv = nws ? provenanceOf(nws, `NWS forecast, zone ${sea.zone ?? ""}`.trim()) : null;
    const lanes = sea.lanes;
    const first = lanes.find((l) => !l.date || l.date === today) ?? lanes[0];
    const tidesToday = sea.tides.filter((t) => t.at.startsWith(today)).sort((a, b) => a.at.localeCompare(b.at));
    const next = nextTide(sea.tides, now);
    const tideLine = (tidesToday.length ? tidesToday.filter((t) => !next || t.at >= next.at).slice(0, 2) : next ? [next] : []).map(tideWords).join(", ");
    const line = first ? periodLine(first) : null;
    if (first || tideLine) {
      agenda.push({
        id: "today",
        start: today,
        title: line ? cap1(line) : first ? strip(first.text) : `Tide: ${tideLine}`,
        detail: [first && line ? `NWS ${first.name.toLowerCase()}` : null, tideLine && (first || line) ? `Tide ${tideLine}` : null].filter(Boolean).join(". ") || null,
        provenance: nwsProv,
      });
    }
    for (const a of sea.alerts) {
      const f = model.today.facts.find((x) => x.id.startsWith("weather:alert:") && x.headline.includes(a.event));
      agenda.push({ id: `alert-${a.event}`, start: today, end: a.ends ? pacificDay(a.ends) : null, title: `NWS ${a.event}`, detail: strip(a.headline), provenance: f ? provenanceOf(f, `NWS ${a.event}`) : null });
    }
    for (const p of comingThatMatter(lanes, first, today)) {
      agenda.push({ id: `nws-${p.date}`, start: p.date!, title: cap1(periodLine(p) ?? strip(p.text)), detail: `NWS forecast for ${p.name}`, provenance: nwsProv });
    }
  }
  agenda.push(...later);

  // ---- soundings (at most two, at their places)
  const soundings: SoundingDatum[] = [];
  if (test?.latest && testHasPlace) {
    const max = maxOf(test);
    const n = test.latest.values_ppm.length;
    const avg = test.latest.avg_ppm;
    const tissue = test.latest.tissue === "unknown" ? "" : ` ${test.latest.tissue}`;
    soundings.push({
      id: "test",
      lon: test.site.lon!,
      lat: test.site.lat!,
      label: `${speciesShort(test.site.species)}${tissue}, highest of ${n}`,
      value: max == null ? "ND" : ppmWords(max),
      unit: test.action != null ? `of ${test.action} ppm` : "ppm",
      caption: [`CDPH, ${day(test.latest.date, { ref: now })}, ${ageWords(test.latest.date, now)}`],
      short: `${speciesShort(test.site.species)} test, CDPH, ${day(test.latest.date, { ref: now })}`,
      stale: test.stale,
      spoken: `${speciesShort(test.site.species)} test at ${test.site.site_label}: highest ${max == null ? "not detected" : `${ppmWords(max)} ppm`}${test.action != null ? ` of the ${test.action} ppm action level` : ""}${avg != null ? `, average ${ppmWords(avg)}` : ""}. Measured by CDPH, ${day(test.latest.date, { ref: now })}, ${ageWords(test.latest.date, now)}. Open where this comes from.`,
      provenance: testProvenance(test),
    });
  }
  const b = sea.buoy;
  if (b && b.lat != null && b.lon != null && b.at && (b.waveFt != null || b.windKt != null)) {
    const f = model.today.facts.find((x) => x.id === "sea:buoy");
    const ageMin = (Date.parse(now) - Date.parse(b.at)) / 60000;
    const waves = b.waveFt != null;
    const value = waves ? b.waveFt!.toFixed(1) : String(Math.round(b.windKt!));
    const unit = waves ? `ft${b.periodS != null ? ` at ${Math.round(b.periodS)} s` : ""}` : `kt${b.windDir ? ` ${b.windDir}` : ""}`;
    const when = `${day(b.at, { ref: now })}, ${new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit" }).format(new Date(b.at))} PT`;
    soundings.push({
      id: "buoy",
      lon: b.lon,
      lat: b.lat,
      label: `${waves ? "Waves" : "Wind"}, buoy ${b.id}`,
      value,
      unit,
      caption: [`NDBC, ${when}`],
      short: `Buoy ${b.id}, ${when.replace(/ PT$/, "")}`,
      stale: ageMin > 180,
      spoken: `${waves ? "Waves" : "Wind"} measured at NDBC buoy ${b.id}: ${value} ${unit}, ${when}. Open where this comes from.`,
      provenance: f ? provenanceOf(f, `Buoy ${b.id}, ${b.name}`, { place: `${Math.round(b.km)} km from ${o.place.name}`, stale: ageMin > 180 }) : null,
    });
  }

  // ---- rows
  const rows: RowDatum[] = [];
  const w = model.whale;
  if (w && isDungeness(o.fishery)) {
    const a = w.action ?? "";
    const short = /trap[s]? prohibited|trap prohibition/i.test(a) ? "traps prohibited" : w.depth_limit_fathoms != null ? `${w.depth_limit_fathoms} fathom limit` : w.gear_reduction_pct != null ? `traps cut ${w.gear_reduction_pct}%` : /fleet advisory/i.test(a) ? "Fleet Advisory" : a ? "measures on record" : "no action on record";
    const parts = a.split(/;\s*/).map(strip).filter(Boolean);
    rows.push({
      id: "whale",
      label: "Whale",
      state: "ok",
      value: `Zone ${w.zone}, ${short}`,
      detail: [{ text: `${parts.map(cap1).join(". ")}.` }, { text: `Zone ${w.zone}: ${w.label}. RAMP is about whale entanglement, not domoic acid.`, provenance: provenanceOf(w.fact, `Whale measures, RAMP zone ${w.zone}`) }],
    });
  }
  const moreTests = model.tests.filter((t) => t !== test && t.latest);
  if (model.testsError) rows.push({ id: "tests", label: "Tests", state: "unavailable", note: "CDPH test results could not be read just now." });
  else if (!model.tests.length) rows.push({ id: "tests", label: "Tests", state: "ok", value: "No CDPH test site for this port", detail: [{ text: "CDPH publishes domoic acid results by site. None of its sites is tied to this port in Neritic's data." }] });
  else if (moreTests.length)
    rows.push({ id: "tests", label: "Tests", state: moreTests.some((t) => t.stale) ? "stale" : "ok", value: `${moreTests.length} more ${moreTests.length === 1 ? "site" : "sites"} nearby`, detail: moreTests.map((t) => ({ text: t.sentence, provenance: testProvenance(t) })) });
  if (sea.state === "error") rows.push({ id: "sea", label: "Sea", state: "unavailable", note: "NWS, the buoy and the tide could not be read just now. Check weather.gov." });
  else if (sea.state === "ok") {
    if (sea.water) {
      const f = model.today.facts.find((x) => x.id === "sea:water");
      rows.push({ id: "water", label: "Water", state: sea.water.stale ? "stale" : "ok", value: `${Math.round((sea.water.c * 9) / 5 + 32)} °F, ${sea.water.where}`, detail: [{ text: `${sea.water.c.toFixed(1)} °C, measured ${sea.water.where}.`, provenance: f ? provenanceOf(f, "Water temperature") : null }] });
    }
    if (sea.windBuoy) {
      const f = model.today.facts.find((x) => x.id === "sea:wind");
      rows.push({ id: "wind", label: "Wind", state: "ok", value: `${sea.windBuoy.windDir ? `${sea.windBuoy.windDir} ` : ""}${Math.round(sea.windBuoy.windKt)} kt, buoy ${sea.windBuoy.id}`, detail: [{ text: f?.detail ?? "", provenance: f ? provenanceOf(f, `Wind, buoy ${sea.windBuoy.id}`) : null }] });
    }
    if (!sea.buoy && sea.buoyReason) rows.push({ id: "buoy", label: "Buoy", state: "unavailable", note: "No buoy reading near this port just now." });
  }
  const models = modelsOf(model);
  rows.push({ id: "models", label: "Models", state: "ok", value: models.value, detail: [] });
  const othersList = model.status
    .filter((r) => r.fishery !== o.fishery)
    .map((r): OtherFishery => ({ fishery: r.fishery, label: r.label, word: r.conflict ? "Status unclear" : r.word, headline: rowHeadline(r), flagged: r.severity === "red" || r.severity === "amber", provenance: r.level === "unknown" ? null : provenanceOf(r.fact, r.label, { value: r.word }) }));

  return {
    place: { key: o.place.key, name: o.place.name, lat: inp.place.lat, lon: inp.place.lon },
    now,
    fishery: o.fishery,
    statement,
    decision,
    agenda,
    sea: sea.state,
    soundings,
    rows,
    models,
    others: othersList,
    allFacts: model.today.facts,
    dataAt: inp.dataGeneratedAt,
    model,
  };
}

/** The other fisheries in one line: "Anchovy & sardine restricted, 6 more". */
export function othersLine(list: OtherFishery[]): string {
  const flagged = list.filter((x) => x.flagged);
  if (!list.length) return "";
  if (!flagged.length) return `${list.length} more, none flagged`;
  const first = flagged[0];
  const rest = list.length - 1;
  return `${first.label.replace(/\s*\(.*\)$/, "")}: ${first.word.toLowerCase()}${rest ? `, ${rest} more` : ""}`;
}

/**
 * NWS periods after today worth a line of their own: an advisory day, or seas or wind well above today's
 * (seas 3 ft higher and at least 8 ft, or wind reaching 20 kt). NWS's words decide; Neritic does not judge.
 * At most two, daytime first.
 */
export function comingThatMatter(lanes: LanePeriod[], first: LanePeriod | undefined, today: string): LanePeriod[] {
  const base = first?.seasHi ?? 0;
  const seen = new Set<string>();
  const out: LanePeriod[] = [];
  for (const p of lanes) {
    if (!p.date || p.date <= today || seen.has(p.date)) continue;
    const big = (p.seasHi != null && p.seasHi >= Math.max(8, base + 3)) || (p.windHi != null && p.windHi >= 20) || /advisory|warning/i.test(p.text);
    if (!big) continue;
    seen.add(p.date);
    out.push(p);
    if (out.length === 2) break;
  }
  return out;
}

/** C-HARM (named and qualified, T9–T12, past run said as past, T3) and the 8-day outlook (algae, not toxin, T5). */
function modelsOf(model: TodayModel): ModelsView {
  const b = model.bloom;
  const c = b.charmFact;
  const charm = c
    ? {
        headline: b.charmPast ? c.headline.replace(/^C-HARM past run/, "Past run of C-HARM") : c.headline,
        words: "P(pDA > 500 ng/L): a probability of conditions in the water, not toxin in seafood. Validation shows near-constant high probabilities near shore.",
        past: b.charmPast || b.charm?.state === "historical",
        provenance: provenanceOf(c, "C-HARM, NOAA model", { stale: b.charm?.state === "stale" || b.charm?.state === "historical" }),
      }
    : null;
  let outlook: ModelsView["outlook"] = null;
  let value: string;
  if (b.outlook && b.outlookFact) {
    const r = b.outlookRead;
    const signal = r ? (r.level === "usual" ? "none" : r.level) : "unknown";
    const head = r && r.level === "usual" ? `No signal: the 8-day chlorophyll outlook is about usual for this time of year.` : b.outlookFact.headline;
    outlook = {
      headline: head,
      words: [b.outlookHonest ?? "Chlorophyll is algae, not toxin.", b.outlookUse].filter((x): x is string => !!x),
      provenance: provenanceOf(b.outlookFact, "Chlorophyll outlook, 8 days", { stale: b.outlookStale }),
      signal,
    };
    value = signal === "none" ? "No signal" : signal === "higher" ? "Algae higher than usual, not toxin" : signal === "lower" ? "Algae lower than usual, not toxin" : "Algae outlook, not toxin";
    if (b.outlookStale) value = "Algae outlook past its dates";
  } else value = charm ? (charm.past ? "Past C-HARM run only" : "C-HARM model run") : "No model run nearby";
  return { value, charm, outlook };
}

/** The fisheries Neritic has something on at this port (a place-specific record or a CDPH test site): a suggestion, labelled as one. */
export function suggestFisheries(inp: TodayInputs): FisheryId[] {
  const key = inp.place.place_id;
  const out = new Set<FisheryId>();
  for (const s of inp.biotoxin?.sites ?? []) {
    if (!s.informs_places.includes(key)) continue;
    const sp = s.species.toLowerCase();
    if (/dungeness/.test(sp)) out.add("dungeness_commercial");
    else if (/rock crab/.test(sp)) out.add("rock_crab");
    else if (/anchov|sardine/.test(sp)) out.add("anchovy_sardine");
    else if (/mussel|clam|oyster/.test(sp)) out.add("bivalve_shellfish");
  }
  for (const r of inp.fisheries?.records ?? []) if (r.applies_to_places.includes(key) && r.fishery !== "other") out.add(r.fishery);
  const order = FISHERY_OPTIONS.map((f) => f.id);
  return [...out].sort((a, b) => order.indexOf(a) - order.indexOf(b));
}
