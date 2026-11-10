import { readPortRecord } from "@/lib/portRecord";

/** The reader's home place as My Coast keeps it on this device (a port, site or point key). Never written here. */
export function readHomePlace(): string | null {
  return readPortRecord().place;
}

/** The home port's CDFW port code, when the home place is a harbour. */
export function readHomePort(): number | null {
  const k = readHomePlace();
  const code = k?.startsWith("port-") ? Number(k.slice(5)) : NaN;
  return Number.isFinite(code) ? code : null;
}
