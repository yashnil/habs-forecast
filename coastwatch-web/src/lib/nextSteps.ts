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
 * Rules, not advice: every step points to an official source or states a gap in what CoastWatch
 * knows. No step says waters are open, seafood is safe, or fishing will be good. Kinds:
 * - "official": an agency notice, statement or regulation page (the authority);
 * - "conditions": NWS/NDBC ocean conditions (context for planning, not a toxin forecast);
 * - "gap": something CoastWatch does not know here, with where to find out.
 */
export type StepKind = "official" | "conditions" | "gap";
export type Step = {
  id: string;
  kind: StepKind;
  title: string;
  reason: string;
  /** the date or time that matters (effective date, issue time, sample date) */
  when?: string;
  source: { label: string; url: string } | { label: string; tel: string };
};

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
};

const fits = (a: Association, audience: Audience) => audience === "all" || AUDIENCE[a.record.fishery].groups.includes(audience);

export function nextSteps(x: StepInput): Step[] {
  const out: Step[] = [];
  const named = (a: Association) => recordNamesSpecies(a.record.species, x.species);
  const src = (a: Association) => a.record.sources.find((s) => s.url.startsWith("https://")) ?? null;

  // 1. notices an agency release appears to lift: read the release first
  for (const a of x.assocs.filter((y) => y.dispute)) {
    out.push({
      id: `dispute-${a.record.id}`,
      kind: "official",
      title: `Check whether “${a.record.title}” is still in effect`,
      reason: `${a.dispute!.agency} release ${a.dispute!.release} appears to lift it; CoastWatch has not reviewed that yet.`,
      source: { label: `${a.dispute!.agency} ${a.dispute!.release}`, url: a.dispute!.url },
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
      title: all.length === 1 ? `Read the notice that applies here: ${first.record.title}` : `Read the ${all.length} notices that apply here, in the agencies' own words`,
      reason: limit ?? (all.length === 1 ? first.basis : all.map((a) => a.record.title).join("; ") + "."),
      when: all.length === 1 && first.record.effective_date ? `since ${x.fmt(first.record.effective_date)}` : undefined,
      source: s ? { label: `${first.record.agency} source`, url: s.url } : { label: AGENCY_LINKS.cdfwAdvisories.label, url: AGENCY_LINKS.cdfwAdvisories.url },
    });
  }

  // 3. notices whose geography does not settle it
  for (const a of x.assocs.filter((y) => y.tier === "unresolved" && !y.dispute && fits(y, x.audience) && named(y)).slice(0, 2)) {
    const s = src(a);
    if (!s) continue;
    out.push({ id: `confirm-${a.record.id}`, kind: "official", title: `Confirm whether “${a.record.title}” covers your waters`, reason: a.basis, source: { label: `${a.record.agency} source`, url: s.url } });
  }

  // 4. CDFW statewide statements about this species, quoted
  if (x.species.statementTopic && x.official) {
    // statements for the chosen activity (a commercial fisher sees the commercial statement), plus season status
    const forActivity = (topic: string) =>
      x.audience === "all" || x.audience === "shellfish" || !/\b(commercial|recreational)\b/i.test(topic) || topic.toLowerCase().includes(x.audience);
    for (const st of x.official.registry.statements.filter((s) => s.topic.toLowerCase().includes(x.species.statementTopic!) && forActivity(s.topic))) {
      out.push({ id: `statement-${st.id}`, kind: "official", title: `${st.agency}: “${st.statement}”`, reason: st.topic, source: { label: `${st.agency} page`, url: st.source.url } });
    }
  }

  // 5. the regulations for this species and region
  if (x.species.link) out.push({ id: "species-rules", kind: "official", title: `Check the rules for ${speciesPhrase(x.species)}`, reason: "Seasons, size and bag limits, and gear rules are set by CDFW and can change during a season.", source: x.species.link });
  if (x.audience !== "commercial") {
    for (const r of cdfwRegionsAt(x.lat).at)
      out.push({ id: `region-${r.id}`, kind: "official", title: `Check CDFW's ${r.name} region sport fishing rules`, reason: `These waters are in the ${r.name} region (${r.limits}).`, source: { label: `CDFW ${r.name} region`, url: r.url } });
  } else {
    out.push({ id: "commercial", kind: "official", title: "Check CDFW commercial fishing rules and permits", reason: "Commercial seasons, permits and closures are set by CDFW.", source: { label: AGENCY_LINKS.cdfwCommercial.label, url: AGENCY_LINKS.cdfwCommercial.url } });
  }
  if (x.audience === "commercial" && /crab/.test(x.species.id))
    out.push({ id: "closure-line", kind: "official", title: "Call CDFW's domoic acid fishery closure line", reason: "Closures and reopenings can be announced before this list is updated.", source: { label: AGENCY_LINKS.cdfwClosureLine.phone, tel: AGENCY_LINKS.cdfwClosureLine.tel } });
  if (x.audience === "shellfish" || ["mussel", "bivalve", "razor-clam"].includes(x.species.id))
    out.push({ id: "cdph-map", kind: "official", title: "Check CDPH's shellfish advisory map, or call the biotoxin line", reason: `CDPH advisories are by county and change with test results; the line is ${AGENCY_LINKS.cdphLine.phone}.`, source: { label: AGENCY_LINKS.cdphMap.label, url: AGENCY_LINKS.cdphMap.url } });
  out.push({ id: "mpa", kind: "official", title: "Check Marine Protected Areas near your waters", reason: "Take is restricted or prohibited inside MPAs; CoastWatch does not draw them.", source: { label: AGENCY_LINKS.cdfwMpa.label, url: AGENCY_LINKS.cdfwMpa.url } });

  // 6. ocean conditions (context for the trip)
  const c = x.conditions;
  if (c?.alerts.status === "ok" && c.alerts.data.length) {
    const a = c.alerts.data[0];
    out.push({ id: "marine-alert", kind: "conditions", title: `${a.event} in effect`, reason: a.headline, when: a.ends ? `until ${x.fmt(a.ends)}` : undefined, source: { label: c.alerts.source, url: c.alerts.url } });
  }
  if (c?.marine.status === "ok") {
    const p = c.marine.data.periods[0];
    out.push({ id: "marine", kind: "conditions", title: `${p.name}: ${marineHeadline(p)}`, reason: `NWS forecast for ${c.marine.data.zoneName} (${c.marine.data.zone}).`, when: c.marine.data.updated ? `issued ${x.fmt(c.marine.data.updated)}` : undefined, source: { label: "NWS marine forecast", url: c.marine.url } });
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
      title: "No recent toxin sample near these waters",
      reason: x.lastToxin ? `The nearest toxin sample within 40 km was taken at ${x.lastToxin.station} (${x.lastToxin.km < 10 ? x.lastToxin.km.toFixed(1) : Math.round(x.lastToxin.km)} km).` : "No station within 40 km measures toxin.",
      when: x.lastToxin ? `last ${x.fmt(x.lastToxin.date)}` : undefined,
      source: { label: AGENCY_LINKS.cdphResults.label, url: AGENCY_LINKS.cdphResults.url },
    });
  }

  // 8. the list itself is not verified: one place to say it, with what to do
  if (x.verification && x.verification.state !== "verified")
    out.push({ id: "verify", kind: "gap", title: "Confirm on the agency pages before you go", reason: "CoastWatch's notice list is transcribed from agency pages and not checked by a person; it can miss or lag a change.", source: { label: AGENCY_LINKS.cdfwAdvisories.label, url: AGENCY_LINKS.cdfwAdvisories.url } });

  return out;
}
