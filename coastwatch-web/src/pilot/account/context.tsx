"use client";

import { createContext, useContext, type ReactNode } from "react";
import type { PilotAccountConfig } from "./config";
import { NOT_CONFIGURED } from "./config";
import { useSession, type Session } from "./hooks";

/**
 * Optional: pages that mount pilot account components (Feedback, AlertsBell, TripLogForm) can wrap
 * them in <PilotAccountProvider config={pilotAccountConfig()}> (server-computed) so they also sync to
 * the signed-in account. Without the provider they work on this device only.
 */
type Ctx = { config: PilotAccountConfig; session: Session };

const PilotAccount = createContext<Ctx | null>(null);

export function PilotAccountProvider({ config, children }: { config: PilotAccountConfig; children: ReactNode }) {
  const session = useSession(config.supabase);
  return <PilotAccount.Provider value={{ config, session }}>{children}</PilotAccount.Provider>;
}

const OFF: Ctx = { config: NOT_CONFIGURED, session: { client: null, user: null, status: "off" } };

export function usePilotAccount(): Ctx {
  return useContext(PilotAccount) ?? OFF;
}
