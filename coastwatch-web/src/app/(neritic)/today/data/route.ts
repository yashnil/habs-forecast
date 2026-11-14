import { NextResponse } from "next/server";
import { isPortKey } from "@/lib/portRecord";
import { loadToday } from "@/pilot/today/load";
import { noticeBands } from "@/nx/today/bands.server";

/**
 * GET /today/data?place=port-N → the published inputs Today builds its page from (src/pilot/today/load.ts):
 * status records, the port's CDPH test sites, notices, the C-HARM cells and the outlook around the port.
 * `/` cannot know the reader's port on the server (it stays on the device, no cookie), so Today at `/`
 * reads this after hydration. Each part that is missing is `null` and the page says so.
 */
export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const place = new URL(req.url).searchParams.get("place");
  if (!isPortKey(place)) return NextResponse.json({ ok: false, error: "place must be a port key (port-N)" }, { status: 400 });
  const l = await loadToday(place).catch((e: Error) => ({ ok: false as const, error: e.message }));
  if (!l.ok) return NextResponse.json({ ok: false, error: l.error }, { status: 503 });
  if (!l.inputs) return NextResponse.json({ ok: false, error: "no such port" }, { status: 404 });
  const bands = await noticeBands((l.inputs.notices ?? []).map((n) => n.id));
  return NextResponse.json({ ok: true, inputs: l.inputs, fixture: l.fixture, bands }, { headers: { "Cache-Control": "public, s-maxage=300, stale-while-revalidate=300" } });
}
