"use client";

import { useEffect, useState } from "react";
import type { Conditions } from "@/lib/conditions";
import type { TodayInputs } from "./model";

/** the official latitude band of each listed notice that has one (bands.server.ts) */
export type Bands = Record<string, { south: number; north: number }>;

/**
 * Today's two reads. The published inputs arrive with the page on /today (server-rendered) and from
 * /today/data at `/` (the port is only known on the device). The sea comes from /api/conditions, the same
 * address the offline copy keeps (src/pilot/today/offline.ts), so a saved Today has its sea too.
 */
export type Initial = { key: string; inputs: TodayInputs; fixture: boolean; bands?: Bands } | null;
export type InputsState = { state: "loading" } | { state: "error" } | { state: "ok"; inputs: TodayInputs; fixture: boolean; bands: Bands };

export function useTodayInputs(key: string | null, initial: Initial): InputsState {
  const seeded = initial && initial.key === key ? initial : null;
  const [s, setS] = useState<{ key: string | null; v: InputsState }>(() => ({ key, v: seeded ? { state: "ok", inputs: seeded.inputs, fixture: seeded.fixture, bands: seeded.bands ?? {} } : { state: "loading" } }));
  useEffect(() => {
    if (!key || (seeded && s.key === key && s.v.state === "ok")) return;
    let live = true;
    setS({ key, v: { state: "loading" } });
    fetch(`/today/data?place=${encodeURIComponent(key)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((j: { ok: boolean; inputs: TodayInputs; fixture: boolean; bands?: Bands }) => live && setS({ key, v: j.ok ? { state: "ok", inputs: j.inputs, fixture: j.fixture, bands: j.bands ?? {} } : { state: "error" } }))
      .catch(() => live && setS({ key, v: { state: "error" } }));
    return () => {
      live = false;
    };
    // the seed is read once per key
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  if (s.key !== key) return seeded ? { state: "ok", inputs: seeded.inputs, fixture: seeded.fixture, bands: seeded.bands ?? {} } : { state: "loading" };
  return s.v;
}

export const conditionsUrl = (lat: number, lon: number) => `/api/conditions?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}&periods=12`;

export function useConditions(url: string | null): Conditions | null | "error" {
  const [c, setC] = useState<{ url: string | null; v: Conditions | null | "error" }>({ url: null, v: null });
  useEffect(() => {
    if (!url) return;
    let live = true;
    fetch(url)
      .then((r) => (r.ok ? (r.json() as Promise<Conditions>) : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => live && setC({ url, v: d }))
      .catch(() => live && setC({ url, v: "error" }));
    return () => {
      live = false;
    };
  }, [url]);
  return c.url === url ? c.v : null;
}
