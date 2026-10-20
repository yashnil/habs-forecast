import { NextResponse } from "next/server";
import { associationsAt } from "@/lib/applicability";
import { parseAudience, resolvePlace } from "@/lib/coast";
import { loadData } from "@/lib/data";
import { classifyTime } from "@/lib/freshness";
import { loadGrid } from "@/lib/grid";
import { measuredEvidence, neighbourhood, stationsAround, EVIDENCE_RADIUS_KM } from "@/lib/hab";
import { harborEvidence, type EvidenceInputs, type SatLite } from "@/lib/harborEvidence";
import { CHARM_LEADS, artifactUrl, charmLayer, charmRun, sourceStatus } from "@/lib/layers";
import { officialVerification } from "@/lib/official";
import { readPlaceChl, satBasis, waterProbe, type WaterProbe } from "@/lib/chlNear";
import { decodeWaterFallback, WATER_FALLBACK_URL } from "@/lib/waterFallback";
import { speciesById } from "@/content/species";
import { speciesStatements } from "@/lib/nextSteps";
import { pacificToday } from "@/lib/time";
import type { Conditions } from "@/lib/conditions";

export const dynamic = "force-dynamic";

/**
 * GET /api/evidence?place=port-550&for=commercial&species=any&date=2026-10-15
 *
 * The harbor evidence graph (lib/harborEvidence) for one place: typed nodes, gaps and claims,
 * built from Neritic's own published data with the same readers My Coast uses. No third-party
 * request is made here; live conditions are not included (they come from /api/conditions, which
 * the page merges). Meant for the review page, for tests, and as the tool surface a future
 * assistant would read, so it never says more than the page does.
 */
export async function GET(req: Request) {
  const url = new URL(req.url);
  const q = url.searchParams;
  const placeKey = q.get("place");
  if (!placeKey) return NextResponse.json({ error: "place is required (port-<code> or point-<lat>,<lon>)" }, { status: 400 });
  const data = await loadData();
  if (!data.ok) return NextResponse.json({ error: "Neritic data are unavailable" }, { status: 503, headers: { "Cache-Control": "no-store" } });
  const place = resolvePlace(placeKey, data.ports);
  if (!place) return NextResponse.json({ error: "unknown place" }, { status: 404 });
  const now = process.env.CW_FIXED_NOW ? new Date(process.env.CW_FIXED_NOW) : new Date();
  const today = pacificToday(now);
  const tripDate = q.get("date") ?? today;
  // grids are fetched from this server's own data path (relative bases cannot be fetched from node)
  const base = /^https?:\/\//.test(data.baseUrl) ? data.baseUrl : `${url.origin}${data.baseUrl}`;

  const measured = {
    cells: measuredEvidence(data.stations, place.lat, place.lon, "cells", now),
    toxin: measuredEvidence(data.stations, place.lat, place.lon, "toxin", now),
    nearest: stationsAround(data.stations, place.lat, place.lon, now, EVIDENCE_RADIUS_KM)[0] ?? null,
  };

  let model: EvidenceInputs["model"] = null;
  try {
    const run = charmRun(data.manifest);
    const days = await Promise.all(
      CHARM_LEADS.map(async (lead) => {
        const a = charmLayer(data.manifest, "particulate_domoic", lead);
        const b = charmLayer(data.manifest, "pseudo_nitzschia", lead);
        const [ca, cb] = await Promise.all([a?.grid ? loadGrid(artifactUrl(base, a.grid.url)) : null, b?.grid ? loadGrid(artifactUrl(base, b.grid.url)) : null]);
        return { lead, valid: (a ?? b)?.time.valid_date ?? null, pda: a?.grid && ca ? neighbourhood(a.grid, ca, place.lat, place.lon) : null, pn: b?.grid && cb ? neighbourhood(b.grid, cb, place.lat, place.lon) : null };
      }),
    );
    const l0 = charmLayer(data.manifest, "particulate_domoic", 0) ?? charmLayer(data.manifest, "pseudo_nitzschia", 0);
    model = run ? { days: days.filter((d) => d.valid), issued: run.issued_date, fresh: l0 ? classifyTime(l0.freshness, l0.time, now) : null } : null;
  } catch {
    model = "error";
  }

  let sat: SatLite | "error" = null;
  try {
    // the same water mask the page applies (shore-adjacent pixels set aside): the bundled water, fetched by absolute URL
    let water: WaterProbe | null = null;
    try {
      const r = await fetch(`${url.origin}${WATER_FALLBACK_URL}`, { signal: AbortSignal.timeout(8000) });
      if (r.ok) water = waterProbe(decodeWaterFallback(await r.json()));
    } catch {
      water = null;
    }
    const pc = await readPlaceChl(data.manifest, base, place.lat, place.lon, water);
    if (pc) {
      const sensor = pc.layer.layer_id.startsWith("olci300") ? "Sentinel-3 OLCI, 300 m" : "VIIRS, 750 m";
      const r = pc.read;
      if (r.kind === "none" || r.kind === "flagged") sat = { value: null, sensor, windowDays: pc.layer.composite?.window_days ?? 7, flagged: r.kind === "flagged", radiusKm: r.radiusKm };
      else sat = { value: r.value, date: r.date, km: r.kind === "at" ? 0 : r.nearest.km, sensor, layerId: pc.layer.layer_id, basis: satBasis(r), aside: r.aside };
    }
  } catch {
    sat = "error";
  }

  const audience = parseAudience(q.get("for"));
  const species = speciesById(q.get("species") ?? "any");
  const official = data.official
    ? { assocs: associationsAt(data.official, place.key, place.lat, place.lon, undefined, place.kind === "point" ? "water" : "shore"), verification: officialVerification(data.official, sourceStatus(data.manifest, "official"), now), audience, species, statements: speciesStatements(data.official, species, audience) }
    : null;
  // live conditions through this server's own route (the same reading the page makes), unless asked not to
  let conditions: Conditions | null = null;
  if (q.get("conditions") !== "0") {
    try {
      const r = await fetch(`${url.origin}/api/conditions?lat=${place.lat}&lon=${place.lon}`, { signal: AbortSignal.timeout(15000) });
      if (r.ok) conditions = (await r.json()) as Conditions;
    } catch {
      conditions = null;
    }
  }
  const modelBeyond = !!(model && model !== "error" && !model.days.some((d) => d.valid === tripDate) && tripDate > (model.days.at(-1)?.valid ?? ""));
  const graph = harborEvidence({ place: { name: place.name, lat: place.lat, lon: place.lon }, now, today, tripDate, measured, measuredError: data.stationsError, model, modelBeyond, sat, conditions, official });
  return NextResponse.json(graph, { headers: { "Cache-Control": "no-store" } });
}
