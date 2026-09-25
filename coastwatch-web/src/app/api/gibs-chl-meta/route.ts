import { NextResponse } from "next/server";
import {
  GIBS_SATELLITE,
  GIBS_PACE_CHL,
  parseViirsNoaa20ChlDefaultDate,
  parsePaceChlDefaultDate,
  fallbackSatelliteDate,
} from "@/lib/gibs";

/** Avoid baking a stale date at build time; cache NASA response briefly at runtime. */
export const dynamic = "force-dynamic";

function fallbackPayload(source: string) {
  const fb = fallbackSatelliteDate(2);
  return {
    date: fb,
    viirsDate: fb,
    paceDate: fb,
    layerId: GIBS_SATELLITE.layerId,
    paceLayerId: GIBS_PACE_CHL.layerId,
    tileMatrixSet: GIBS_SATELLITE.tileMatrixSet,
    source,
  };
}

export async function GET() {
  try {
    const res = await fetch(GIBS_SATELLITE.wmtsCapsUrl, {
      next: { revalidate: 3600 },
      headers: { Accept: "application/xml" },
    });
    if (!res.ok) {
      return NextResponse.json(fallbackPayload("fallback_http_error"));
    }
    const xml = await res.text();
    const viirs = parseViirsNoaa20ChlDefaultDate(xml);
    const pace = parsePaceChlDefaultDate(xml);
    const v = viirs ?? fallbackSatelliteDate(2);
    const p = pace ?? v;
    return NextResponse.json({
      date: v,
      viirsDate: v,
      paceDate: p,
      layerId: GIBS_SATELLITE.layerId,
      paceLayerId: GIBS_PACE_CHL.layerId,
      tileMatrixSet: GIBS_SATELLITE.tileMatrixSet,
      source: viirs ? "gibs_getcapabilities" : "fallback_parse",
    });
  } catch {
    return NextResponse.json(fallbackPayload("fallback_exception"));
  }
}
