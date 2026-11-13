"use client";

import { useEffect, useMemo, useState } from "react";
import { buildToday } from "@/nx/today/model";
import { useTodayInputs } from "@/nx/today/data";
import { zoneCaption } from "@/nx/today/ticket";
import { sectorOf, whaleMeasures } from "@/nx/regulation/whale";
import { useOnline } from "@/nx/offline/useOffline";

/**
 * The whale-zone plate's second line on Alerts ("Season delayed · Fleet advisory"), in the port's records' words,
 * derived exactly as Today derives it (src/nx/today/TodayView.tsx) for commercial Dungeness, the fishery the RAMP
 * zones govern. Null while the records load, offline, or when they say nothing.
 */
export function useZoneCaption(port: { key: string; name: string } | null): string | null {
  // offline nothing is fetched (Alerts checks nothing without a connection): the plate keeps its zone name only
  // (and not before hydration, when the online state is still the server's guess)
  const online = useOnline();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const ask = mounted && online && typeof navigator !== "undefined" && navigator.onLine !== false;
  const s = useTodayInputs(ask ? (port?.key ?? null) : null, null);
  const inputs = s.state === "ok" ? s.inputs : null;
  return useMemo(() => {
    if (!inputs || !port) return null;
    try {
      const fishery = "dungeness_commercial" as const;
      const v = buildToday(inputs, { fishery, sea: null, place: port });
      const whale = v.model.whale;
      if (!whale) return null;
      const st = v.statement;
      const whaleDelay = st.state === "ok" && st.row.cause === "whale_entanglement";
      return zoneCaption(whaleDelay && st.state === "ok" ? st.level : null, whaleMeasures(whale, sectorOf(fishery), { seasonSaysIt: whaleDelay }).short.split(", ")[0]);
    } catch {
      return null;
    }
  }, [inputs, port?.key, port?.name]); // eslint-disable-line react-hooks/exhaustive-deps
}
