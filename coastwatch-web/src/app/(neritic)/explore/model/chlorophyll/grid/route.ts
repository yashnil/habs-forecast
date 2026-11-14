import { NextResponse } from "next/server";
import { chlField, chlSource } from "@/lib/ocean/chlorophyll";
import type { ChlVar } from "@/lib/ocean/contract";
import { errMsg } from "@/lib/ocean/upstream";

export const dynamic = "force-dynamic";

const VARS: ChlVar[] = ["chl", "mask"];

/**
 * GET /explore/model/chlorophyll/grid?date=2026-11-02&var=chl|mask → one packed grid (uint16 little-endian, gzip,
 * log10 for chl; geometry and scale are in the index). A day's field does not change, so it is cached for a day.
 */
export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const date = q.get("date") ?? "", v = q.get("var") as ChlVar;
  if (!VARS.includes(v)) return NextResponse.json({ status: "unavailable", reason: "var must be chl or mask", source: chlSource().source }, { status: 400 });
  try {
    const packed = await chlField(date);
    return new NextResponse(new Uint8Array(packed[v]), {
      headers: { "Content-Type": "application/octet-stream", "Cache-Control": "public, max-age=3600, s-maxage=86400" },
    });
  } catch (e) {
    const bad = e instanceof RangeError;
    return NextResponse.json({ status: "unavailable", reason: bad ? errMsg(e) : `${chlSource().short} could not be read: ${errMsg(e)}`, source: chlSource().source }, { status: bad ? 400 : 503, headers: { "Cache-Control": "no-store" } });
  }
}
