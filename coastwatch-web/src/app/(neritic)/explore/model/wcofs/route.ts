import { NextResponse } from "next/server";
import { wcofsIndex } from "@/lib/ocean/wcofs";

export const dynamic = "force-dynamic";

/**
 * GET /explore/model/wcofs → the NOAA WCOFS index (lib/ocean/contract WcofsIndex): the newest cycle, its steps and
 * valid times, grid geometry and grid URLs, labels and caveats. Read live from CO-OPS THREDDS with caching; nothing
 * is published. An outage with nothing cached answers status "unavailable".
 */
export async function GET() {
  const body = await wcofsIndex();
  const cache = body.status === "unavailable" ? "public, s-maxage=120" : "public, s-maxage=900, stale-while-revalidate=900";
  return NextResponse.json(body, { headers: { "Cache-Control": cache } });
}
