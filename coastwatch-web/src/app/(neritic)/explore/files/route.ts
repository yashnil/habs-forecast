import { NextResponse } from "next/server";
import { isPortKey } from "@/lib/portRecord";
import { gridFiles } from "@/nx/explore/chunks";
import { loadExploreModel } from "@/nx/explore/load.server";
import { LAYER_LABEL } from "@/nx/explore/words";
import { shellPorts } from "@/nx/shell/ports.server";

export const dynamic = "force-dynamic";

/**
 * GET /explore/files?place=port-N → what a field pack keeps of Explore for one port (src/nx/offline/pack.ts):
 * each layer's latest frame, as the files that cover the port's area (a chunked statewide grid only by the
 * chunks around the port), with the frame's own stamp so the pack can say what it holds. Nothing is added
 * to the published manifest; this only reads it.
 */
export async function GET(req: Request) {
  const place = new URL(req.url).searchParams.get("place");
  if (!isPortKey(place)) return NextResponse.json({ ok: false, error: "place must be a port key" }, { status: 400 });
  const port = (await shellPorts()).find((p) => p.key === place);
  if (!port) return NextResponse.json({ ok: false, error: "unknown port" }, { status: 404 });
  const l = await loadExploreModel();
  if (!l.ok) return NextResponse.json({ ok: false, error: "the published data could not be read" }, { status: 503 });
  const m = l.model;
  // the port's Explore view at any screen shape, with room to pan a little
  const box: [number, number, number, number] = [port.lon - 1.3, port.lat - 0.9, port.lon + 0.6, port.lat + 0.9];
  const base = m.baseUrl.replace(/\/$/, "");
  const frames = m.products.flatMap((p) => {
    const f = p.frames.find((x) => x.id === p.defaultFrame);
    if (!f?.grid) return [];
    const files = [...gridFiles(f.grid, box), ...gridFiles(f.ageGrid, box), ...gridFiles(f.typicalGrid, null)];
    return [{ product: p.id, layer: LAYER_LABEL[p.id], frame: f.id, label: f.label, stamp: f.stamp, files: files.map((x) => `${base}/${x}`) }];
  });
  return NextResponse.json({ ok: true, frames }, { headers: { "Cache-Control": "no-store" } });
}
