import type { LayerArtifact, Manifest } from "@/generated/schema";

/**
 * Multi-sensor display rule, mirrored from the pipeline (sources/multisensor.py `pick`):
 * Sentinel-3 (primary) where it has an observation, unless VIIRS (secondary) observed that
 * place more than `tolDays` days more recently; VIIRS where Sentinel-3 has none; else none.
 * Each pixel shows one sensor's own value and date; nothing is averaged.
 */
export type SensorObs = { value: number; date: string } | null;
export type Pick = 0 | 1 | 2;

const DAY = 86_400_000;
const ord = (d: string) => Date.parse(`${d}T00:00:00Z`) / DAY;

export function multiSensorPick(primary: SensorObs, secondary: SensorObs, tolDays: number): Pick {
  if (primary && !(secondary && ord(secondary.date) > ord(primary.date) + tolDays)) return 1;
  return secondary ? 2 : 0;
}

/** Observation date of a latest-clear-view pixel from its age in days. */
export function observedDate(referenceDate: string, ageDays: number): string {
  return new Date(Date.parse(`${referenceDate}T00:00:00Z`) - ageDays * DAY).toISOString().slice(0, 10);
}

export function multiSensorLayer(m: Manifest): LayerArtifact | null {
  return m.layers.find((l) => l.layer_id === "multisensor_chl_latest" && l.multisensor) ?? null;
}

/** The member layers of a multi-sensor display, primary first. */
export function multiSensorMembers(m: Manifest, l: LayerArtifact | null): [LayerArtifact | null, LayerArtifact | null] {
  const ms = l?.multisensor;
  if (!ms) return [null, null];
  const sorted = [...ms.members].sort((a, b) => a.order - b.order);
  const find = (id: string | undefined) => m.layers.find((x) => x.layer_id === id) ?? null;
  return [find(sorted[0]?.layer_id), find(sorted[1]?.layer_id)];
}

/** "Sentinel-3 read 35% lower than VIIRS" from a median log10 ratio. */
export function ratioPhrase(medianLog10: number, primary = "Sentinel-3", secondary = "VIIRS"): string {
  const f = 10 ** medianLog10;
  if (Math.abs(f - 1) < 0.05) return `${primary} and ${secondary} read about the same`;
  return f < 1 ? `${primary} read about ${Math.round((1 - f) * 100)}% lower than ${secondary}` : `${primary} read about ${Math.round((f - 1) * 100)}% higher than ${secondary}`;
}
