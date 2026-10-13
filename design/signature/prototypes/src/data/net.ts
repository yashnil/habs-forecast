/**
 * Simulated network conditions for the prototypes, set from the URL so a reviewer can reproduce a
 * state with a link:
 *   ?net=slow          every resource waits 0.6–3.5 s (staggered, deterministic per resource)
 *   ?net=3g            every resource waits 1.5–6 s
 *   ?fail=charm,official,observations,ports,satellite,currents,tiles,fonts
 *   ?cache=1           failed resources fall back to a "cached" copy (the same snapshot, flagged)
 * Nothing here changes data values; failures are reported as failures.
 */
export type ResourceKey =
  | "manifest" | "charm" | "official" | "observations" | "ports" | "portIntel" | "satellite" | "currents" | "tiles" | "fonts";

const params = new URLSearchParams(typeof location !== "undefined" ? location.hash.split("?")[1] ?? location.search : "");
export const NET = {
  profile: (params.get("net") ?? "fast") as "fast" | "slow" | "3g",
  fail: new Set((params.get("fail") ?? "").split(",").filter(Boolean) as ResourceKey[]),
  cache: params.get("cache") === "1",
};

export function refreshNetFromUrl() {
  const p = new URLSearchParams(location.hash.split("?")[1] ?? "");
  NET.profile = (p.get("net") ?? "fast") as typeof NET.profile;
  NET.fail = new Set((p.get("fail") ?? "").split(",").filter(Boolean) as ResourceKey[]);
  NET.cache = p.get("cache") === "1";
}

const ORDER: ResourceKey[] = ["manifest", "official", "ports", "observations", "portIntel", "charm", "satellite", "currents", "tiles", "fonts"];
export function latencyFor(key: ResourceKey): number {
  const i = ORDER.indexOf(key);
  if (NET.profile === "slow") return 600 + ((i * 397) % 2900);
  if (NET.profile === "3g") return 1500 + ((i * 811) % 4500);
  return 0;
}

export class SourceError extends Error {
  constructor(public key: ResourceKey, public reason: "unreachable" | "http" | "decode", message: string) {
    super(message);
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function fetchResource(key: ResourceKey, url: string, init?: RequestInit): Promise<Response> {
  const wait = latencyFor(key);
  if (wait) await sleep(wait);
  if (NET.fail.has(key)) throw new SourceError(key, "unreachable", `${key}: simulated failure (?fail=${key})`);
  const res = await fetch(url, init);
  if (!res.ok) throw new SourceError(key, "http", `${key}: HTTP ${res.status}`);
  return res;
}

export async function fetchJSON<T>(key: ResourceKey, url: string): Promise<T> {
  return (await fetchResource(key, url)).json() as Promise<T>;
}
