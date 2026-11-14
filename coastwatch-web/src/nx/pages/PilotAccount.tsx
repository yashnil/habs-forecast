"use client";

import type { ReactNode } from "react";
import type { PilotAccountConfig } from "@/pilot/account/config";
import { PilotAccountProvider } from "@/pilot/account/context";
import { ReadinessProvider } from "@/v4/readiness/store";

/** The providers the pilot's account page reads (they came with the pilot's v4 shell before v15.1). */
export function PilotAccountFrame({ config, children }: { config: PilotAccountConfig; children: ReactNode }) {
  return (
    <PilotAccountProvider config={config}>
      <ReadinessProvider>{children}</ReadinessProvider>
    </PilotAccountProvider>
  );
}
