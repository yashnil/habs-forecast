import { NextResponse } from "next/server";
import { chlEstimateIndex } from "@/lib/ocean/chlorophyll";

export const dynamic = "force-dynamic";

/**
 * GET /explore/model/chlorophyll → the gap-filled chlorophyll estimate index (lib/ocean/contract ChlEstimateIndex):
 * the newest day with its real date and age, a stale flag, grid geometry and URLs, label and caveats. Read live from
 * NOAA CoastWatch ERDDAP with a 6 h cache; nothing is published.
 */
export async function GET() {
  const body = await chlEstimateIndex();
  const cache = body.status === "unavailable" ? "public, s-maxage=120" : "public, s-maxage=3600, stale-while-revalidate=3600";
  return NextResponse.json(body, { headers: { "Cache-Control": cache } });
}
