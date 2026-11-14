"use client";

import "@/nx/ds/tokens.css";
import "@/nx/chart/chart.css";
import "@/nx/shell/shell.css";
import { Suspense, type ReactNode } from "react";
import { MotionPrefs } from "@/nx/ds/MotionPrefs";
import { Workspace } from "@/nx/shell/Margin";
import { PortsProvider, type ShellPort } from "@/nx/shell/port";
import { Shell } from "@/nx/shell/Shell";

/**
 * The shell without the route group's layout, for the root's not-found and error pages (an address outside the group,
 * a failure above it): the same bar, menu and tab bar, the page on the informational layout, no chart behind it.
 */
export function StandaloneShell({ ports = [], children }: { ports?: ShellPort[]; children: ReactNode }) {
  return (
    <div className="n12-app" data-testid="nx-app">
      <MotionPrefs />
      <PortsProvider value={ports}>
        {/* the bar reads the address (the port); a not-found page is prerendered, so it waits for the browser */}
        <Suspense fallback={null}>
          <Shell />
        </Suspense>
        <Workspace chart={null}>{children}</Workspace>
      </PortsProvider>
    </div>
  );
}
