import { loadConditions } from "@/lib/conditionsServer";
import { buildTodayModel } from "@/pilot/facts";
import { conditionsPoint, TODAY_PERIODS } from "@/pilot/today/portToday";
import { loadToday } from "@/pilot/today/load";
import { gatesFrom, seaFrom, type CoastError, type CoastGates, type CoastSea } from "./brief";
import { tripFishery } from "./fisheries";

/**
 * Server side of My Coast's margin. The port lives on the reader's device (nx-port-v1), so the page asks for
 * it by key: the gates come from the published dataset (fast), the sea from NWS, NDBC and CO-OPS (live,
 * a separate request so the gates never wait on it). Both go through Today's loaders and model.
 */
export async function coastGates(placeKey: string, fisheryId: string): Promise<CoastGates | CoastError | null> {
  const f = tripFishery(fisheryId);
  if (!f) return { ok: false, error: "unknown fishery" };
  const load = await loadToday(placeKey).catch((e: Error) => ({ ok: false as const, error: e.message }));
  if (!load.ok) return { ok: false, error: load.error };
  if (!load.place || !load.inputs) return null;
  const m = buildTodayModel({ ...load.inputs, selected: [f.id], conditions: null });
  return gatesFrom(m, f, { now: load.inputs.now, fixture: load.fixture });
}

export async function coastSea(placeKey: string): Promise<CoastSea | CoastError | null> {
  const load = await loadToday(placeKey).catch((e: Error) => ({ ok: false as const, error: e.message }));
  if (!load.ok) return { ok: false, error: load.error };
  if (!load.place || !load.inputs) return null;
  const p = conditionsPoint(load.inputs.place.lat, load.inputs.place.lon);
  const conditions = await loadConditions(p.lat, p.lon, TODAY_PERIODS).catch(() => "error" as const);
  const m = buildTodayModel({ ...load.inputs, selected: [], conditions });
  return seaFrom(m, load.inputs.now);
}
