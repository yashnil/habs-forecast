"use client";

import { useEffect } from "react";
import type { PortToday } from "@/pilot/contracts";
import { useAccount } from "@/pilot/account/hooks";
import { runLocalCheck } from "./inbox";

/**
 * Renders nothing: compares the given PortToday facts with what this device saw last time and adds
 * in-app alerts for the person's saved ports and triggers. Today mounts it with the port's facts for
 * every fishery (the same facts /account and the fan-out compare):
 *     <AlertCheck todays={[today]} sample={fixture} />
 * Idempotent (one alert per fact version), tolerant of blocked storage. `sample`: the facts are the
 * test fixture's, and any alert made from them says so.
 */
export function AlertCheck({ todays, sample = false }: { todays: PortToday[]; sample?: boolean }) {
  const { account, hydrated } = useAccount();
  const saved = account.saved_places.join(",");
  const prefs = JSON.stringify(account.alerts);
  const facts = todays.map((t) => `${t.place_id}@${t.generated_at}`).join(",");
  useEffect(() => {
    if (hydrated && todays.length) runLocalCheck(account, todays, new Date(), { sample });
    // keyed by content, not identity: an inline [today] does not re-run it
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, facts, saved, prefs]);
  return null;
}
