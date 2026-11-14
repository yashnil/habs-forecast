import { NextResponse } from "next/server";
import { buoyObservation } from "@/lib/conditionsServer";

export const dynamic = "force-dynamic";

/**
 * GET /explore/buoy?id=46042 → that NDBC station's newest observations (wind, waves, water temperature), each with
 * its own time (lib/conditionsServer buoyObservation). Explore California Waters reads it when a buoy is opened;
 * nothing is published, and a station that has not reported says so.
 */
export async function GET(req: Request) {
  const id = new URL(req.url).searchParams.get("id") ?? "";
  if (!/^[A-Z0-9]{4,8}$/i.test(id)) return NextResponse.json({ status: "none", reason: "id must be an NDBC station id", source: "NOAA National Data Buoy Center" }, { status: 400 });
  const body = await buoyObservation(id.toUpperCase());
  return NextResponse.json(body, { headers: { "Cache-Control": "public, s-maxage=600, stale-while-revalidate=300" } });
}
