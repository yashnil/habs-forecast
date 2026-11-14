import { NextResponse } from "next/server";
import { clampPeriods, loadConditions, validCoastPoint } from "@/lib/conditionsServer";

/**
 * GET /api/conditions?lat=…&lon=…&periods=… → ocean conditions for one place (lib/conditionsServer.ts:
 * NWS, NDBC, CO-OPS, CeNCOOS; each part fails on its own). Responses are cached for 10 minutes.
 */
export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const lat = Number(q.get("lat"));
  const lon = Number(q.get("lon"));
  if (!validCoastPoint(lat, lon)) {
    return NextResponse.json({ error: "lat and lon must be California coastal coordinates" }, { status: 400 });
  }
  // `periods`: how many NWS forecast periods to return (4 by default; Today asks for 12, about five days)
  const body = await loadConditions(lat, lon, clampPeriods(Number(q.get("periods") ?? 4)));
  return NextResponse.json(body, { headers: { "Cache-Control": "public, s-maxage=600, stale-while-revalidate=300" } });
}
