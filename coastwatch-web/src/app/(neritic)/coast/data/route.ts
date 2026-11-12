import { NextResponse } from "next/server";
import { isPortKey } from "@/lib/portRecord";
import { coastGates, coastSea } from "@/nx/coast/brief.server";

export const dynamic = "force-dynamic";

/**
 * GET /coast/data?place=port-592&fishery=dungeness_commercial → My Coast's gates for one fishery at one port.
 * GET /coast/data?place=port-592&part=sea                    → its sea window (NWS, NDBC, CO-OPS; slower).
 * The port is the reader's, kept on their device, so the page asks by key; nothing about the reader is stored.
 */
export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const place = q.get("place");
  if (!isPortKey(place)) return NextResponse.json({ ok: false, error: "place must be a port key" }, { status: 400 });
  const body = q.get("part") === "sea" ? await coastSea(place) : await coastGates(place, q.get("fishery") ?? "");
  if (body === null) return NextResponse.json({ ok: false, error: "unknown port" }, { status: 404 });
  const status = body.ok ? 200 : body.error === "unknown fishery" ? 400 : 503;
  return NextResponse.json(body, { status, headers: { "Cache-Control": q.get("part") === "sea" ? "public, s-maxage=600, stale-while-revalidate=300" : "no-store" } });
}
