import type { OfficialDataset } from "@/generated/official";
import { AGENCY_LINKS } from "@/content/coast";
import { recordNamesSpecies, speciesPhrase, type Species } from "@/content/species";
import { AUDIENCE, cdfwRegionsAt, type Association } from "@/lib/applicability";
import type { Audience } from "@/lib/coast";
import { compass, marineHeadline, type Conditions } from "@/lib/conditions";
import type { Verification } from "@/lib/official";

/**
 * "Next steps" for a place, an activity and a species: what to read or check before going,
 * each with the reason it is suggested, the time that matters, and the source to open.
 *
 * Rules, not advice: every step points to an official source or states a gap in what Neritic
 * knows. No step says waters are open, seafood is safe, or fishing will be good. Kinds:
 * - "official": an agency notice, statement or regulation page (the authority);
 * - "conditions": NWS/NDBC ocean conditions (context for planning, not a toxin forecast);
 * - "gap": something Neritic does not know here, with where to find out;
 * - "hab": a recent shore measurement near these waters, to read at its source (seawater, not seafood);
 * - "history": the species' landings history (history, never an outlook).
 * Each step says why it is suggested here, which waters or species it concerns, and how sure it is.
 */
export type StepKind = "official" | "conditions" | "gap" | "hab" | "history";
export type Step = {
  id: string;
  kind: StepKind;
  title: string;
  reason: string;
  /** the date or time that matters (effective date, issue time, sample date) */
  when?: string;
  source: { label: string; url: string } | { label: string; tel: string };
  /** the waters and species the step concerns */
  scope?: string;
  /** how sure Neritic is about it, in plain words */
  caveat?: string;
  /** a second source to read alongside (a disputed notice: the original notice) */
  also?: { label: string; url: string };
};

/** A recent shore sample near the waters (CalHABMAP), as the briefing shows it. */
export type NearSample = { kind: "toxin" | "cells"; station: string; stationId: string; km: number; date: string; ageDays: number; text: string };

export type StepInput = {
  official: OfficialDataset | null;
  verification: Verification | null;
  assocs: Association[];
  audience: Audience;
  species: Species;
  lat: number;
  conditions: Conditions | null;
  /** latest toxin (pDA) sample within 40 km, if any */
  lastToxin: { station: string; km: number; date: string; ageDays: number } | null;
  /** formats a date or time for display */
  fmt: (iso: string) => string;
  /** recent shore samples within the briefing's radius (newest first), if any */
  samples?: NearSample[];
};

/** How the landings groups read inside a sentence (the group, not the chosen species, is what the history covers). */
const GROUP_PHRASE: Record<string, string> = {
  dungeness_crab: "Dungeness crab",
  rock_crab: "rock crab",
  northern_anchovy: "northern anchovy",
  spiny_lobster: "spiny lobster",
  bivalves: "bivalve shellfish",
};

/** The species' landings group on the Fisheries page (history only). */
export const FISHERIES_GROUP: Partial<Record<string, string>> = {
  dungeness: "dungeness_crab",
  "rock-crab": "rock_crab",
  "anchovy-sardine": "northern_anchovy",
  lobster: "spiny_lobster",
  mussel: "bivalves",
  bivalve: "bivalves",
  "razor-clam": "bivalves",
};

/**
 * CDFW's statewide statements about the chosen species for the chosen activity, a fishery's season
 * status first (it says more about whether anyone may fish than "no toxin closures" does).
 */
export function speciesStatements(official: OfficialDataset | null, species: Species, audience: Audience) {
  if (!species.statementTopic || !official) return [];
  const forActivity = (topic: string) => audience === "all" || audience === "shellfish" || !/\b(commercial|recreational)\b/i.test(topic) || topic.toLowerCase().includes(audience);
  return official.registry.statements
    .filter((s) => s.topic.toLowerCase().includes(species.statementTopic!) && forActivity(s.topic))
    .sort((a, b) => Number(/season status/i.test(b.topic)) - Number(/season status/i.test(a.topic)));
}

const fits = (a: Association, audience: Audience) => audience === "all" || AUDIENCE[a.record.fishery].groups.includes(audience);

export function nextSteps(x: StepInput): Step[] {
  const out: Step[] = [];
  const named = (a: Association) => recordNamesSpecies(a.record.species, x.species);
  const src = (a: Association) => a.record.sources.find((s) => s.url.startsWith("https://")) ?? null;
  const who = x.species.id === "any" ? "any species" : speciesPhrase(x.species);
  const activity = { all: "", commercial: "commercial ", recreational: "recreational ", shellfish: "shellfish " }[x.audience];

  // 1. notices an agency release appears to lift: read the release first
  for (const a of x.assocs.filter((y) => y.dispute && named(y))) {
    out.push({
      id: `dispute-${a.record.id}`,
      kind: "official",
      title: `Check whether “${a.record.title}” is still in effect`,
      reason: `${a.dispute!.agency} release ${a.dispute!.release} (“${a.dispute!.title}”) appears to lift it. Read the release and the original notice, and confirm with ${a.dispute!.agency} before relying on either.`,
      source: { label: `${a.dispute!.agency} ${a.dispute!.release}, the later release`, url: a.dispute!.url },
      also: src(a) ? { label: `the original notice`, url: src(a)!.url } : undefined,
      scope: `${a.record.species.join(", ")}${a.record.area.description ? ` · ${a.record.area.description}` : ""}`,
      caveat: "No person has reviewed the release yet, so Neritic shows the notice as neither in effect nor lifted.",
    });
  }

  // 2. notices that apply here by their official geography (listed, with sources, in the official card)
  const applying = x.assocs.filter((a) => a.tier === "geographic" && !a.dispute && fits(a, x.audience) && named(a) && a.record.area.type !== "statewide");
  const statewide = x.assocs.filter((a) => a.tier === "geographic" && !a.dispute && fits(a, x.audience) && named(a) && a.record.area.type === "statewide");
  const all = [...applying, ...statewide];
  if (all.length) {
    const first = all[0];
    const s = src(first);
    const limit = all.flatMap((a) => a.caveats).find((c) => /limit/.test(c));
    out.push({
      id: "read-notices",
      kind: "official",
      title: all.length === 1 ? `Read the notice that covers these waters: ${first.record.title}` : `Read the ${all.length} notices that cover these waters, in the agencies' own words`,
      reason: limit ?? (all.length === 1 ? first.basis : all.map((a) => a.record.title).join("; ") + "."),
      when: all.length === 1 && first.record.effective_date ? `since ${x.fmt(first.record.effective_date)}` : undefined,
      source: s ? { label: `${first.record.agency} source`, url: s.url } : { label: AGENCY_LINKS.cdfwAdvisories.label, url: AGENCY_LINKS.cdfwAdvisories.url },
      scope: all.length === 1 ? `${first.record.species.join(", ")}${first.record.area.description ? ` · ${first.record.area.description}` : ""}` : x.species.id === "any" ? `${x.audience === "all" ? "Any" : activity.charAt(0).toUpperCase() + activity.slice(1)}${x.audience === "all" ? " activity" : x.audience === "shellfish" ? "harvest" : "fishing"} in these waters` : `${activity}${who} in these waters`,
      caveat: "Tied to these waters by the notice's own geography; the notices are transcribed, not verified, and no person has reviewed the association.",
    });
  }

  // 3. notices whose geography does not settle it
  for (const a of x.assocs.filter((y) => y.tier === "unresolved" && !y.dispute && fits(y, x.audience) && named(y)).slice(0, 2)) {
    const s = src(a);
    if (!s) continue;
    out.push({
      id: `confirm-${a.record.id}`,
      kind: "official",
      title: `Confirm whether “${a.record.title}” covers your waters`,
      reason: a.basis,
      source: { label: `${a.record.agency} source`, url: s.url },
      scope: a.record.species.join(", "),
      caveat: "Its official geography does not settle whether it covers these waters; only the agency can.",
    });
  }

  // 4. CDFW statewide statements about this species, quoted
  {
    // statements for the chosen activity (a commercial fisher sees the commercial statement), season status first
    for (const st of speciesStatements(x.official, x.species, x.audience)) {
      out.push({ id: `statement-${st.id}`, kind: "official", title: `${st.agency}: “${st.statement}”`, reason: st.topic, source: { label: `${st.agency} page`, url: st.source.url }, scope: `${speciesPhrase(x.species)}, statewide`, caveat: "Quoted from the agency's page as last checked; it can change without this list noticing." });
    }
  }

  // 5. the regulations for this species and region
  // a recent shore sample near these waters: read it where it was published
  const sample = (x.samples ?? []).find((m) => m.ageDays <= 30);
  if (sample)
    out.push({
      id: "hab-sample",
      kind: "hab",
      title: `Look at the latest sample at ${sample.station}: ${sample.text}`,
      reason: `${sample.kind === "cells" ? "Pseudo-nitzschia can produce domoic acid; a count says the algae were there, not that seafood is toxic." : "Particulate domoic acid in seawater; seafood can differ."} The station's history shows whether this is unusual for it.`,
      when: `sampled ${x.fmt(sample.date)}`,
      source: { label: `${sample.station} on Bloom`, url: `/bloom?station=${encodeURIComponent(sample.stationId)}` },
      scope: `Seawater at ${sample.station}, ${sample.km < 10 ? sample.km.toFixed(1) : Math.round(sample.km)} km from these waters`,
      caveat: "One sample at one shore point; blooms change within days and differ offshore.",
    });

  // Dungeness crab: the season can also be held or limited for whales and sea turtles (RAMP), which
  // CDFW publishes only as prose; Neritic links it and infers nothing from it
  if (x.species.id === "dungeness")
    out.push({
      id: "whale-safe",
      kind: "official",
      title: "Check CDFW's whale and sea turtle (RAMP) actions for your zone",
      reason: "CDFW can delay the season or limit trap depth to protect whales and sea turtles; this is separate from toxin testing.",
      source: { label: "CDFW Whale Safe Fisheries", url: "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries" },
      scope: `${activity.charAt(0).toUpperCase() + activity.slice(1)}Dungeness crab, by fishing zone`,
      caveat: "Neritic does not track these actions; only CDFW's page and declarations say what applies.",
    });
  else if (x.species.link) out.push({ id: "species-rules", kind: "official", title: `Check the rules for ${speciesPhrase(x.species)}`, reason: "Seasons, size and bag limits, and gear rules are set by CDFW and can change during a season.", source: x.species.link, scope: `${activity.charAt(0).toUpperCase() + activity.slice(1)}${activity ? speciesPhrase(x.species) : x.species.label}` });
  if (x.audience !== "commercial") {
    for (const r of cdfwRegionsAt(x.lat).at)
      out.push({ id: `region-${r.id}`, kind: "official", title: `Check CDFW's ${r.name} region sport fishing rules`, reason: `These waters are in the ${r.name} region (${r.limits}).`, source: { label: `CDFW ${r.name} region`, url: r.url } });
  } else {
    out.push({ id: "commercial", kind: "official", title: "Check CDFW commercial fishing rules and permits", reason: "Commercial seasons, permits and closures are set by CDFW.", source: { label: AGENCY_LINKS.cdfwCommercial.label, url: AGENCY_LINKS.cdfwCommercial.url } });
  }
  if (x.audience === "commercial" && /crab|dungeness/.test(x.species.id))
    out.push({ id: "closure-line", kind: "official", title: "Call CDFW's domoic acid fishery closure line", reason: "Closures and reopenings can be announced before this list is updated.", source: { label: AGENCY_LINKS.cdfwClosureLine.phone, tel: AGENCY_LINKS.cdfwClosureLine.tel }, scope: `Commercial ${speciesPhrase(x.species)}` });
  if (x.audience === "shellfish" || ["mussel", "bivalve", "razor-clam"].includes(x.species.id))
    out.push({
      id: "cdph-map",
      kind: "official",
      title: "Check CDPH's shellfish advisory map, or call the biotoxin line",
      reason: `CDPH tests shellfish meat for domoic acid and paralytic shellfish toxins; its advisories are by county and change with results. The line is ${AGENCY_LINKS.cdphLine.phone}.`,
      source: { label: AGENCY_LINKS.cdphMap.label, url: AGENCY_LINKS.cdphMap.url },
      also: { label: "CDPH's latest test results (PDF)", url: AGENCY_LINKS.cdphResults.url },
      scope: `${x.species.id === "any" ? "Shellfish" : x.species.label} in this county`,
      caveat: "Neritic does not carry CDPH's shellfish results; seawater measurements are not a substitute.",
    });
  out.push({ id: "mpa", kind: "official", title: "Check Marine Protected Areas near your waters", reason: "Take is restricted or prohibited inside MPAs; Neritic does not draw them.", source: { label: AGENCY_LINKS.cdfwMpa.label, url: AGENCY_LINKS.cdfwMpa.url } });

  // the species' landings history (Fisheries): history, never an outlook
  const group = FISHERIES_GROUP[x.species.id];
  if (group)
    out.push({
      id: "history",
      kind: "history",
      title: `See California's ${GROUP_PHRASE[group] ?? speciesPhrase(x.species)} landings history`,
      reason: "Landings by year from public fisheries records, with the 2015–16 domoic acid closures marked; how much of any change the closures explain is not stated.",
      source: { label: "Fisheries history", url: `/fisheries?group=${group}` },
      scope: `${GROUP_PHRASE[group]!.charAt(0).toUpperCase() + GROUP_PHRASE[group]!.slice(1)}, statewide`,
      caveat: "History only: it says nothing about this season or these waters.",
    });

  // 6. ocean conditions (context for the trip)
  const c = x.conditions;
  if (c?.alerts.status === "ok" && c.alerts.data.length) {
    const a = c.alerts.data[0];
    out.push({ id: "marine-alert", kind: "conditions", title: `${a.event} in effect`, reason: a.headline, when: a.ends ? `until ${x.fmt(a.ends)}` : undefined, source: { label: c.alerts.source, url: c.alerts.url } });
  }
  if (c?.marine.status === "ok") {
    const p = c.marine.data.periods[0];
    out.push({ id: "marine", kind: "conditions", title: `${p.name}: ${marineHeadline(p)}`, reason: `NWS forecast for ${c.marine.data.zoneName} (${c.marine.data.zone}).`, when: c.marine.data.updated ? `issued ${x.fmt(c.marine.data.updated)}` : undefined, source: { label: "NWS marine forecast", url: c.marine.url } });
  } else if (c && c.marine.status === "none" && c.marine.url) {
    out.push({ id: "marine-link", kind: "conditions", title: "Read NWS's marine forecast and any warnings for your waters", reason: "Wind, waves and small craft advisories come from the National Weather Service; Neritic links to them rather than copying them.", source: { label: "NWS marine forecasts", url: c.marine.url } });
  } else if (c && c.marine.status === "unavailable") {
    out.push({ id: "marine-gap", kind: "gap", title: "Check the NWS marine forecast directly", reason: `NWS's forecast service did not answer just now (${c.marine.reason.replace(/^The NWS forecast did not load /, "").replace(/[()]/g, "").trim()}).`, source: { label: "NWS marine forecasts", url: "https://www.weather.gov/marine/" } });
  }
  if (c?.buoy.status === "ok" && c.buoy.data.windKt && c.buoy.station) {
    const b = c.buoy.data;
    const windAt = c.buoy.data.windKt.at;
    const parts = [b.windKt ? `wind ${Math.round(b.windKt.value)} kt${b.windDirDeg ? ` ${compass(b.windDirDeg.value)}` : ""}` : null, b.waveFt ? `waves ${b.waveFt.value.toFixed(1)} ft` : null].filter(Boolean);
    out.push({ id: "buoy", kind: "conditions", title: `Measured at ${c.buoy.station.name}: ${parts.join(", ")}`, reason: `NDBC station ${c.buoy.station.id}, ${Math.round(c.buoy.station.km)} km away. A single station, not the whole area.`, when: `observed ${x.fmt(windAt)}`, source: { label: `NDBC ${c.buoy.station.id}`, url: c.buoy.url } });
  }

  // 7. gaps: no recent toxin sample
  if (!x.lastToxin || x.lastToxin.ageDays > 30) {
    out.push({
      id: "toxin-gap",
      kind: "gap",
      title: x.audience === "shellfish" ? "No recent seawater toxin sample near these waters; shellfish testing is CDPH's" : "No recent toxin sample near these waters",
      reason: x.lastToxin ? `The nearest toxin sample within 40 km was taken at ${x.lastToxin.station} (${x.lastToxin.km < 10 ? x.lastToxin.km.toFixed(1) : Math.round(x.lastToxin.km)} km).` : "No station within 40 km measures toxin.",
      when: x.lastToxin ? `last ${x.fmt(x.lastToxin.date)}` : undefined,
      source: { label: AGENCY_LINKS.cdphResults.label, url: AGENCY_LINKS.cdphResults.url },
      caveat: "No sample is not a clean result.",
    });
  }

  // 8. the list itself is not verified: one place to say it, with what to do
  if (x.verification && x.verification.state !== "verified")
    out.push({ id: "verify", kind: "gap", title: "Confirm on the agency pages before you go", reason: "Neritic's notice list is transcribed from agency pages and not checked by a person; it can miss or lag a change.", source: { label: AGENCY_LINKS.cdfwAdvisories.label, url: AGENCY_LINKS.cdfwAdvisories.url } });

  // shellfish harvesters read CDPH's results before anything else that is not a notice
  if (x.audience === "shellfish" || ["mussel", "bivalve", "razor-clam"].includes(x.species.id)) {
    const i = out.findIndex((y) => y.id === "cdph-map");
    const at = out.findIndex((y) => !/^(dispute|read-notices|confirm)/.test(y.id));
    if (i > 0 && at >= 0 && at < i) out.splice(at, 0, out.splice(i, 1)[0]);
  }
  return out;
}
