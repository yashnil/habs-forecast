"use client";

import { useEffect } from "react";
import type { PortToday } from "@/pilot/contracts";
import { useAccount } from "@/pilot/account/hooks";
import { runLocalCheck } from "./inbox";

/**
 * Renders nothing: compares the given PortToday facts with what this device saw last time and adds
 * in-app alerts for the person's saved ports and triggers. Mount it on the Today page:
 *     <AlertCheck todays={[today]} />
 * Idempotent (one alert per fact version), tolerant of blocked storage.
 */
export function AlertCheck({ todays }: { todays: PortToday[] }) {
  const { account, hydrated } = useAccount();
  const saved = account.saved_places.join(",");
  const prefs = JSON.stringify(account.alerts);
  const facts = todays.map((t) => `${t.place_id}@${t.generated_at}`).join(",");
  useEffect(() => {
    if (hydrated && todays.length) runLocalCheck(account, todays);
    // keyed by content, not identity: an inline [today] does not re-run it
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, facts, saved, prefs]);
  return null;
}
