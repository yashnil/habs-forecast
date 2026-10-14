import { NextResponse } from "next/server";
import { buildSnapshot } from "@/lib/briefingSnapshot";
import { parseAudience } from "@/lib/coast";
import { loadData } from "@/lib/data";

export const dynamic = "force-dynamic";

/**
 * GET /api/briefing?place=port-201&for=commercial&species=dungeness
 *
 * The briefing snapshot for one place, activity and species (lib/briefingSnapshot), built from
 * CoastWatch's own published data. It makes no request to any third-party service. Used by
 * "Since you last checked" and kept briefings.
 */
export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const place = q.get("place");
  if (!place) return NextResponse.json({ error: "place is required" }, { status: 400 });
  const data = await loadData();
  if (!data.ok) return NextResponse.json({ error: "CoastWatch data are unavailable" }, { status: 503, headers: { "Cache-Control": "no-store" } });
  const snap = buildSnapshot(data, place, parseAudience(q.get("for")), q.get("species") ?? "any", new Date());
  if (!snap) return NextResponse.json({ error: "unknown place" }, { status: 404 });
  return NextResponse.json(snap, { headers: { "Cache-Control": "no-store" } });
}
