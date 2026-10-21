"use client";

import { useMemo } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { StationLite } from "@/lib/data";
import { associationsAt } from "@/lib/applicability";
import { ageDays, stationsNear, type Audience, type CoastPlace } from "@/lib/coast";
import { CHARM_LEADS, charmLayer, sourceStatus } from "@/lib/layers";
import { classifyTime } from "@/lib/freshness";
import { officialVerification } from "@/lib/official";
import { pacificToday } from "@/lib/time";
import { harborEvidence, pageClaims, type EvidenceGraph } from "@/lib/harborEvidence";
import { fmtMeasured } from "@/lib/hab";
import { nextSteps, speciesStatements, type NearSample } from "@/lib/nextSteps";
import { checklist, parseTripDate, periodFor, tripDays } from "@/lib/trip";
import { CHL_NEAR_COPY } from "@/content/copy";
import type { Species } from "@/content/species";
import { useModel } from "@/components/coast/CoastSections";
import { useHabEvidence, useOutlook, useSatellite } from "@/components/coast/HabBriefing";
import { fmtWhen, officialAnswer, useConditions } from "@/components/coast/Briefing";

/**
 * Everything the trip briefing says about one place, assembled exactly as My Coast (v2) assembled
 * it: the evidence graph (lib/harborEvidence) is the one source of every claim, so the briefing, the
 * Explore inspector and /api/evidence say the same thing. Presentation lives in Briefing.tsx.
 */
export function useBriefing(p: {
  manifest: Manifest;
  ports: PortsCollection | null;
  official: OfficialDataset | null;
  stations: StationLite[] | null;
  stationsError: string | null;
  baseUrl: string;
  portIntel: PortIntelCollection | null;
  place: CoastPlace | null;
  audience: Audience;
  species: Species;
  dateParam: string | null;
  now: Date | null;
}) {
  const { manifest, official, stations, stationsError, baseUrl, place, audience, species, now } = p;
  const verification = useMemo(() => (official && now ? officialVerification(official, sourceStatus(manifest, "official"), now) : null), [official, manifest, now]);
  const assocs = useMemo(() => (place ? associationsAt(official, place.key, place.lat, place.lon, undefined, place.kind === "point" ? "water" : "shore") : []), [official, place]);
  const st = useMemo(() => (place ? stationsNear(stations, place.lat, place.lon) : { near: [], nearestBeyond: null }), [stations, place]);
  const leads = useMemo(() => CHARM_LEADS.map((l) => charmLayer(manifest, "particulate_domoic", l)), [manifest]);
  const today = now ? pacificToday(now) : null;
  const days = useMemo(() => (today ? tripDays(today, leads) : []), [today, leads]);
  const tripDate = parseTripDate(p.dateParam, days);
  const tripDay = days.find((d) => d.date === tripDate) ?? null;
  const issuedLeads = leads.map((l, k) => (l ? k : -1)).filter((k) => k >= 0);
  const lead = tripDay?.lead ?? (issuedLeads.at(-1) ?? 0);
  const model = useModel(manifest, baseUrl, place?.lat ?? null, place?.lon ?? null, lead);
  const cond = useConditions(place?.lat ?? null, place?.lon ?? null);
  const outlook = useOutlook(manifest, baseUrl, place?.lat ?? null, place?.lon ?? null);
  const satNow = useSatellite(manifest, baseUrl, place?.lat ?? null, place?.lon ?? null);
  const hab = useHabEvidence(stations, place?.lat ?? null, place?.lon ?? null, now);
  const outDay = outlook && outlook !== "error" ? (outlook.find((d) => d.lead === lead) ?? outlook[0] ?? null) : null;

  const status = useMemo(() => speciesStatements(official, species, audience), [official, species, audience]);
  const runLayer = charmLayer(manifest, "particulate_domoic", 0) ?? charmLayer(manifest, "pseudo_nitzschia", 0);
  const runFresh = runLayer && now ? classifyTime(runLayer.freshness, runLayer.time, now) : null;
  const modelBeyond = !!tripDay && tripDay.lead == null && outlook !== null && outlook !== "error";

  const nowDay = now ? today : null;
  const issuedKey = runLayer?.time.issued_date;
  const runState = runFresh?.state;
  const runAge = runFresh?.ageDays;
  const verState = verification?.state;
  const statusKey = status.map((x) => x.id).join();
  // the graph reads the NWS period for the trip day only, matched by the period's own date (lib/trip
  // periodFor): a saved copy read days later has none, and never says "Tonight" for a past night
  const tripPeriod = periodFor(cond.data, tripDate, today);
  const condForGraph = useMemo(() => {
    const c = cond.data;
    if (!c || c.marine.status !== "ok") return c;
    return { ...c, marine: { ...c.marine, data: { ...c.marine.data, periods: tripPeriod ? [{ name: tripPeriod.name, text: tripPeriod.text }] : [] } } };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cond.data, tripPeriod?.name, tripPeriod?.date]);
  const graph = useMemo<EvidenceGraph | null>(
    () =>
      place && now
        ? harborEvidence({
            place: { name: place.name, lat: place.lat, lon: place.lon },
            now,
            today,
            tripDate,
            measured: hab ? { cells: hab.cells, toxin: hab.toxin, nearest: hab.nearest } : null,
            measuredError: stationsError,
            model: outlook === null ? "loading" : outlook === "error" ? "error" : { days: outlook, issued: runLayer?.time.issued_date ?? null, fresh: runFresh },
            modelBeyond,
            sat: satNow,
            conditions: condForGraph,
            official: { assocs, verification, audience, species, statements: status },
          })
        : null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [place, nowDay, tripDate, hab, stationsError, outlook, issuedKey, runState, runAge, modelBeyond, satNow, condForGraph, assocs, verState, audience, species, statusKey],
  );
  const matters = useMemo(() => (graph ? pageClaims(graph) : []), [graph]);

  // the one count of notices (v6 P0 1): those tied to these waters by official geography (not disputed),
  // which the top bar, the glance and the notices section all use; and of those, the ones the graph
  // keeps for the reader's activity and species. "None here" is never said when any are tied.
  const notices = useMemo(() => {
    const here = assocs.filter((a) => a.tier === "geographic" && !a.dispute);
    const hereIds = new Set(here.map((a) => a.record.id));
    const naming = (graph?.nodes ?? []).filter((n) => n.variable === "official notice" && n.id?.startsWith("official:") && hereIds.has(n.id.slice(9)));
    return {
      here: here.length,
      namingSpecies: naming.length,
      hereAssocs: here,
      namingIds: new Set(naming.map((n) => n.id!.slice(9))),
      unsettled: assocs.filter((a) => a.tier === "unresolved" && !a.dispute).map((a) => a.record.id),
    };
  }, [assocs, graph]);

  const unknowns = useMemo(() => {
    if (!place) return [];
    const out: string[] = [];
    const tied = assocs.filter((a) => a.tier === "geographic" && !a.dispute && a.record.area.type !== "statewide");
    for (const a of assocs.filter((x) => x.dispute)) out.push(`${a.dispute!.agency} release ${a.dispute!.release} appears to lift “${a.record.title}”; no person has reviewed it yet.`);
    if (!tied.length) out.push("No notice in Neritic's list is tied to these waters by its geography. The list may be incomplete; check the agencies.");
    out.push("Whether each notice applies here was worked out from the notices' official geography; no person has reviewed these associations, and the notices themselves are not verified.");
    const nUnresolved = assocs.filter((a) => a.tier === "unresolved" && !a.dispute).length;
    if (nUnresolved) out.push(`Whether ${nUnresolved === 1 ? "one notice applies" : `${nUnresolved} notices apply`} to these waters is not settled by ${nUnresolved === 1 ? "its" : "their"} official geography.`);
    const recentToxin = st.near.some((s) => s.station.latest.some((m) => m.variable === "pDA" && now && ageDays(m.date, now) <= 30));
    if (!recentToxin) out.push("There is no toxin measurement from the last 30 days within 40 km.");
    if (model?.pda && !model.pda.here) out.push(`C-HARM gives no toxin probability for these waters${model.pda.nearest ? `; its nearest value is ${model.pda.nearest.km.toFixed(1)} km away and is not a forecast here` : ""}.`);
    const lyr = model?.layer;
    if (lyr && now && classifyTime(lyr.freshness, lyr.time, now).state !== "current") out.push("The latest C-HARM run is not current.");
    if (satNow !== "error" && satNow && satNow.value == null) out.push(satNow.flagged ? CHL_NEAR_COPY.unknownFlagged : "No clear satellite view of these waters in the last week (cloud, fog or no overpass).");
    out.push("Toxin in seafood is not measured by Neritic. CDPH tests shellfish; paralytic shellfish poisoning (PSP) results are not shown here.");
    return out;
  }, [place, assocs, st, model, satNow, now]);

  const lastToxin = st.near
    .flatMap((s) => s.station.latest.filter((m) => m.variable === "pDA" && m.value != null).map((m) => ({ s, m })))
    .sort((a, b) => b.m.date.localeCompare(a.m.date))[0];
  const toxin = lastToxin && now ? { station: lastToxin.s.station.name, km: lastToxin.s.km, date: lastToxin.m.date, ageDays: ageDays(lastToxin.m.date, now), value: `${Number(lastToxin.m.value!.toPrecision(2))} ${lastToxin.m.units}` } : null;
  const toxinDate = toxin?.date;
  const toxinStation = toxin?.station;
  const samples = useMemo<NearSample[]>(
    () =>
      [hab?.cells, hab?.toxin]
        .filter((e): e is NonNullable<typeof e> => !!e && e.near)
        .sort((a, b) => a.ageDays - b.ageDays)
        .map((e) => ({
          kind: e.kind === "toxin" ? "toxin" : "cells",
          station: e.station.name,
          stationId: e.station.station_id,
          km: e.km,
          date: e.date,
          ageDays: e.ageDays,
          text: e.kind === "cells" ? `Pseudo-nitzschia ${fmtMeasured("pn_seriata", e.values.reduce((a, v) => a + v.value, 0))}` : `domoic acid ${fmtMeasured("pDA", e.values[0].value)}`,
        })),
    [hab],
  );
  const steps = useMemo(
    () => (place ? nextSteps({ official, verification, assocs, audience, species, lat: place.lat, conditions: cond.data, lastToxin: toxin, fmt: fmtWhen, samples }) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [place, official, verification, assocs, audience, species, cond.data, toxinDate, toxinStation, samples],
  );
  const checks = useMemo(() => checklist(steps), [steps]);
  const officialHeadline = officialAnswer(assocs, audience, species);

  return { graph, matters, notices, tripPeriod, unknowns, steps, checks, assocs, verification, days, tripDate, tripDay, lead, model, outlook, outDay, satNow, hab, cond, st, status, officialHeadline, today };
}
