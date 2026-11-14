"use client";

import type { ReactNode } from "react";
import type { OfficialDataset } from "@/generated/official";
import type { SourceStatus } from "@/generated/schema";
import { OfficialDrawer, OfficialProvider } from "@/components/shell/OfficialShell";

/**
 * What Fisheries' M7 components read from the frame they were built in (AppShell): the official notices and their
 * drawer (the species table opens it from a notice count). Inside the token bridge, so the drawer takes the ds colours.
 */
export function FisheriesFrame({ official, status, children }: { official: OfficialDataset | null; status: SourceStatus | null; children: ReactNode }) {
  return (
    <div className="nxp-bridge">
      <OfficialProvider official={official} status={status}>
        {children}
        <OfficialDrawer />
      </OfficialProvider>
    </div>
  );
}
