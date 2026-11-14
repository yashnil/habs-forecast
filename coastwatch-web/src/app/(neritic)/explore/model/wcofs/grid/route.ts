import { NextResponse } from "next/server";
import type { WcofsVar } from "@/lib/ocean/contract";
import { errMsg } from "@/lib/ocean/upstream";
import { SOURCE, wcofsStep } from "@/lib/ocean/wcofs";

export const dynamic = "force-dynamic";

const VARS: WcofsVar[] = ["u", "v", "sst", "mask"];

/**
 * GET /explore/model/wcofs/grid?cycle=2026111403&step=f024&var=u|v|sst|mask → one packed grid (uint16 little-endian,
 * gzip; lib/grid loadGrid decodes it; geometry and scale are in the index). A cycle's step never changes, so it is
 * cached for a day. Upstream failure answers 503 with a JSON reason.
 */
export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const cycle = q.get("cycle") ?? "", step = q.get("step") ?? "", v = q.get("var") as WcofsVar;
  if (!VARS.includes(v)) return NextResponse.json({ status: "unavailable", reason: "var must be u, v, sst or mask", source: SOURCE }, { status: 400 });
  try {
    const packed = await wcofsStep(cycle, step);
    return new NextResponse(new Uint8Array(packed[v]), {
      headers: {
        "Content-Type": "application/octet-stream",
        "Cache-Control": "public, max-age=3600, s-maxage=86400, immutable",
        "X-Valid-Time": packed.valid_time,
      },
    });
  } catch (e) {
    const bad = e instanceof RangeError;
    return NextResponse.json({ status: "unavailable", reason: bad ? errMsg(e) : `WCOFS could not be read: ${errMsg(e)}`, source: SOURCE }, { status: bad ? 400 : 503, headers: { "Cache-Control": "no-store" } });
  }
}
