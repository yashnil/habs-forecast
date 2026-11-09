"use client";

import "@/pilot/today/today.css";
import "./pilot.css";
import Link from "next/link";
import type { ReactNode } from "react";
import { useNow } from "@/lib/useNow";
import { NOT_CONFIGURED, type PilotAccountConfig } from "@/pilot/account/config";
import { PilotAccountProvider } from "@/pilot/account/context";
import { ReadinessProvider } from "@/v4/readiness/store";
import { SkipLink } from "@/v4/shell/SkipLink";
import { NavTabs, TopBar, type Workspace } from "@/v4/shell/TopBar";

/**
 * ONE shell for the pilot: Today, Ask Neritic, Your ports and alerts and the legal pages share the
 * v4 top bar (workspaces, bell, status, ⋯), the phone dock, the "Almanac" ground (today.css,
 * pilot.css; dark by default, daylight from the same toggle) and one footer. Moving between them is
 * moving inside one app: nothing reloads its own chrome or changes its type.
 */

/** The phone dock: the four workspaces, fixed to the bottom edge. */
export function PilotDock({ active }: { active: Workspace }) {
  return (
    <div className="env-instrument fixed inset-x-0 bottom-0 z-40 border-t border-n-line bg-n-surface lg:hidden" style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
      <NavTabs active={active} size="dock" />
    </div>
  );
}

/** The standing disclaimer and the pilot's own pages, under every pilot page. */
export function PilotFooter() {
  return (
    <footer className="mt-10 border-t border-n-line py-6 text-[13px] leading-relaxed text-n-ink-3" data-testid="pilot-footer">
      <p className="max-w-[92ch]">
        Neritic is an independent research preview, not an official source and not a seafood-safety service. Closures, seasons and advisories come from CDFW and CDPH; their notices always take precedence. Chlorophyll does not measure toxin. Forecasts are probabilities, not guarantees.
      </p>
      <nav aria-label="Pilot pages" className="mt-2 flex flex-wrap gap-x-5 text-[13.5px]">
        {(
          [
            ["/today", "Today"],
            ["/ask", "Ask Neritic"],
            ["/account", "Your ports and alerts"],
            ["/privacy", "Privacy"],
            ["/terms", "Terms"],
            ["/sms-terms", "SMS terms"],
            ["/sources", "Sources"],
          ] as const
        ).map(([href, label]) => (
          <Link key={href} href={href} className="nx-focus inline-flex min-h-11 items-center text-n-ink-2 underline decoration-n-line underline-offset-4 hover:text-n-ink" data-testid={`footer-${href.slice(1)}`}>
            {label}
          </Link>
        ))}
      </nav>
    </footer>
  );
}

/** The hero band of a pilot page: a caps line, the serif title, a sentence, then anything the page adds. */
export function PilotHero({ eyebrow, title, titleId, lede, children }: { eyebrow: ReactNode; title: ReactNode; titleId?: string; lede?: ReactNode; children?: ReactNode }) {
  return (
    <header className="today-hero" data-testid="pilot-hero">
      <div className="today-wrap relative pb-6 pt-5 md:pb-9 md:pt-8">
        <p className="nx-caps">{eyebrow}</p>
        <h1 id={titleId} className="pilot-title mt-2 text-n-ink">
          {title}
        </h1>
        {lede && <p className="pilot-lede mt-3">{lede}</p>}
        {children}
      </div>
    </header>
  );
}

export function PilotShell({
  active = "page",
  children,
  testid,
  footer = true,
  account = NOT_CONFIGURED,
}: {
  active?: Workspace;
  children: ReactNode;
  testid?: string;
  /** false: the page ends with its own footer */
  footer?: boolean;
  /** what accounts may do here (pilotAccountConfig(), computed on the server): the bell and feedback then follow the signed-in account */
  account?: PilotAccountConfig;
}) {
  const now = useNow();
  return (
    <PilotAccountProvider config={account}>
    <ReadinessProvider>
      <div className="env-instrument today pilot min-h-dvh pb-[calc(76px+env(safe-area-inset-bottom,0px))] lg:pb-10" data-testid={testid}>
        <SkipLink />
        <TopBar active={active} now={now} />
        <main id="main">
          {children}
          {footer && (
            <div className="today-wrap">
              <PilotFooter />
            </div>
          )}
        </main>
        <PilotDock active={active} />
      </div>
    </ReadinessProvider>
    </PilotAccountProvider>
  );
}
