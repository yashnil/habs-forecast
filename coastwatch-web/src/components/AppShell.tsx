"use client";

import Link from "next/link";
import { MotionConfig } from "motion/react";
import { PageIn } from "@/components/motion/PageIn";
import { RESEARCH } from "@/content/copy";
import { BRAND } from "@/content/brand";
import { OfficialDrawer, OfficialProvider } from "@/components/shell/OfficialShell";
import type { Manifest, SourceStatus } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import { isFixture, sourceStatus } from "@/lib/layers";
import { ReadinessProvider, useReportSource } from "@/v4/readiness/store";
import { Dock, TopBar } from "@/v4/shell/TopBar";
import { useNow } from "@/lib/useNow";
import { StagePoster } from "@/v4/stage/StagePoster";
import "./appshell.css";

/**
 * The frame of the reference pages (Blooms, Fisheries, Data status, About) in v4: the shared top bar
 * (workspaces, status centre, official notices), the page on the light reading environment, the
 * footer, and the dock on phones. The pages' own content is unchanged; inside `.env-chart` the v2
 * tokens resolve to the v4 palette and faces (src/v4/theme.css), so it reads as one system.
 */
export function AppShell({
  children,
  manifest,
  official = null,
}: {
  scroll?: "app" | "page";
  children: React.ReactNode;
  active: "map" | "coast" | "bloom" | "fisheries" | "sources" | "about";
  manifest: Manifest | null;
  official?: OfficialDataset | null;
  officialStatus?: SourceStatus | null;
}) {
  return (
    <MotionConfig reducedMotion="user">
      <ReadinessProvider>
        <OfficialProvider official={official} status={null}>
          <Frame manifest={manifest} official={official}>
            {children}
          </Frame>
          <OfficialDrawer />
        </OfficialProvider>
      </ReadinessProvider>
    </MotionConfig>
  );
}

function Frame({ children, manifest, official }: { children: React.ReactNode; manifest: Manifest | null; official: OfficialDataset | null }) {
  const now = useNow();
  useReportSource({ id: "data", label: "Neritic dataset", state: manifest ? "ready" : "unavailable", detail: manifest ? `Published ${manifest.generated_at.slice(0, 16).replace("T", " ")} UTC` : "Could not be loaded", asOf: manifest?.generated_at ?? null });
  return (
    <div className="env-chart flex min-h-dvh flex-col pb-[72px] text-n-ink lg:pb-0">
      <a href="#main" className="sr-only z-50 rounded-md bg-n-surface px-3 py-2 text-[14px] font-semibold text-n-accent focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:flex focus:min-h-11 focus:items-center" data-testid="skip-link">
        Skip to main content
      </a>
      <TopBar active="page" now={now} trust={{ official, status: manifest ? sourceStatus(manifest, "official") : null, fixture: !!manifest && isFixture(manifest) }} />
      {!manifest && (
        <div className="mx-auto w-full max-w-[1280px] px-4 sm:px-8">
          <p role="alert" className="pb-1 text-[14px] font-semibold text-n-down" data-testid="data-unavailable">
            Neritic&rsquo;s data could not be loaded. Official closures and advisories are always on the CDFW and CDPH sites.
          </p>
        </div>
      )}
      <main id="main" tabIndex={-1} className="relative isolate flex min-h-0 flex-1 flex-col outline-none">
        {/* the quiet head of a reference page: the real Monterey Canyon isobaths (brand geometry, never the reader's waters) */}
        <div className="as-ground" aria-hidden>
          <StagePoster />
        </div>
        <PageIn kind="page">{children}</PageIn>
      </main>
      <footer data-testid="site-footer" className="as-footer">
        <div className="mx-auto flex max-w-[1280px] flex-wrap items-center gap-x-8 gap-y-2 px-4 py-10 text-[14px] text-n-ink-2 sm:px-8">
          <span className="font-semibold text-n-ink" data-testid="footer-brand">
            {BRAND.name} · research preview
          </span>
          <span className="text-n-ink-3" data-testid="footer-disclaimer">
            Not an official source and not independently validated. It does not replace CDFW or CDPH: closures, advisories and seafood safety come from them.
          </span>
          <span className="w-full text-[13px] text-n-ink-3 sm:order-last" data-testid="footer-independence">
            {BRAND.independence}
          </span>
          <nav aria-label={`About ${BRAND.name}`} className="flex flex-wrap gap-x-5 gap-y-2 sm:ml-auto">
            <Link href="/about" className="inline-flex min-h-[44px] min-w-[44px] items-center font-medium underline underline-offset-2">
              About
            </Link>
            <Link href="/sources" className="inline-flex min-h-[44px] items-center font-medium underline underline-offset-2">
              Data &amp; sources
            </Link>
            <a href={RESEARCH.doi} target="_blank" rel="noreferrer" className="inline-flex min-h-[44px] items-center font-medium underline underline-offset-2">
              Research paper ↗
            </a>
          </nav>
        </div>
      </footer>
      <Dock active="page" />
    </div>
  );
}
