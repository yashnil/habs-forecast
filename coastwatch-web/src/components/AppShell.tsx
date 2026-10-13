import Link from "next/link";
import { MotionConfig } from "motion/react";
import { PageIn } from "@/components/motion/PageIn";
import { RESEARCH } from "@/content/copy";
import { ModeSwitch, PrimaryNav } from "@/components/shell/ModeNav";
import { SourceHealthDot } from "@/components/SourceHealthDot";
import { OfficialDrawer, OfficialPill, OfficialProvider, TabBar } from "@/components/shell/OfficialShell";
import { MastheadMenu } from "@/components/shell/MastheadMenu";
import type { Manifest, SourceStatus } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";

/**
 * Page frame (design reset §4): a navy masthead on every page with the primary navigation,
 * the official-notices pill and data status; a bottom tab bar on phones. Reading pages sit
 * on light paper; the map ("app") variant puts its content on the dark sea theme.
 */
export function AppShell({
  children,
  active,
  manifest,
  official = null,
  officialStatus = null,
  banner,
  scroll = "app",
}: {
  /** "app": fixed-height viewport on the dark map theme; "page": the document scrolls, on paper */
  scroll?: "app" | "page";
  children: React.ReactNode;
  active: "map" | "coast" | "bloom" | "fisheries" | "sources" | "about";
  manifest: Manifest | null;
  official?: OfficialDataset | null;
  officialStatus?: SourceStatus | null;
  banner?: React.ReactNode;
}) {
  return (
    // every animation follows the device's reduced-motion setting
    <MotionConfig reducedMotion="user">
    <OfficialProvider official={official} status={officialStatus}>
      {/* keyboard users skip the masthead and the status line */}
      <a
        href="#main"
        className="sr-only z-50 rounded-md bg-surface px-3 py-2 text-[14px] font-medium text-accent shadow focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
        data-testid="skip-link"
      >
        Skip to main content
      </a>
      <div className={`flex min-h-dvh flex-col bg-page pb-[var(--cw-tabbar-h)] md:pb-0 ${scroll === "app" ? "lg:h-dvh" : ""}`}>
        <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-navy-950 text-on-navy">
          <div className="flex h-[52px] min-w-0 items-center gap-3 px-3 min-[360px]:px-4 min-[400px]:gap-6 md:h-[var(--cw-masthead-h)] md:gap-5 md:px-6 lg:gap-8">
            <Link href="/" className="flex shrink-0 items-center gap-2 min-[360px]:gap-2.5" aria-label="CoastWatch home">
              <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
                <rect width="32" height="32" rx="8" fill="#10263e" />
                <path d="M6 19c3.2-2.6 6.4-2.6 9.6 0s6.6 2.6 10.4-.2" fill="none" stroke="#6cc6dc" strokeWidth="2.2" strokeLinecap="round" />
                <path d="M6 13.5c3.2-2.6 6.4-2.6 9.6 0s6.6 2.6 10.4-.2" fill="none" stroke="#e8eef5" strokeOpacity=".55" strokeWidth="2.2" strokeLinecap="round" />
                <circle cx="22.5" cy="8.5" r="2.2" fill="#f6bb5c" />
              </svg>
              <span className="text-[17px] font-semibold tracking-tight text-white max-[359px]:sr-only">CoastWatch</span>
              <span className="hidden border-l border-white/15 pl-2.5 text-[13px] text-on-navy-2 lg:inline">California</span>
            </Link>
            <PrimaryNav active={active} />
            <div className="ml-auto flex items-center gap-2 min-[360px]:gap-3">
              <div className="hidden xl:block">
                <ModeSwitch />
              </div>
              <Link
                href="/about"
                data-testid="nav-about"
                aria-current={active === "about" ? "page" : undefined}
                className={`hidden px-1.5 text-[14px] md:block ${active === "about" ? "text-white" : "text-on-navy-2 hover:text-on-navy"}`}
              >
                About
              </Link>
              <OfficialPill />
              <Link
                href="/sources"
                data-testid="data-status-link"
                aria-current={active === "sources" ? "page" : undefined}
                className={`items-center gap-2 rounded-md px-1.5 py-1.5 text-[14px] max-md:hidden md:flex ${active === "sources" ? "text-white" : "text-on-navy-2 hover:text-on-navy"}`}
              >
                <SourceHealthDot manifest={manifest} />
                <span className="hidden whitespace-nowrap lg:inline">Data status</span>
                <span className="sr-only lg:hidden">Data status</span>
              </Link>
              <MastheadMenu active={active} />
            </div>
          </div>
          {/* the status line is part of the masthead: one navy band, not a second cream bar (signature P1) */}
          {banner}
        </header>
        <main id="main" tabIndex={-1} className={`flex min-h-0 flex-1 flex-col outline-none ${scroll === "app" ? "theme-dark" : ""}`}>
          <PageIn kind={scroll}>{children}</PageIn>
        </main>
        {scroll === "page" && (
          <footer className="border-t border-hairline bg-page" data-testid="site-footer">
            <div className="mx-auto flex max-w-[1280px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-6 text-[14px] text-ink-2 sm:px-8">
              <span className="font-semibold text-ink">CoastWatch</span>
              <span className="text-ink-3">Not an official source. Closures and advisories come from CDFW and CDPH.</span>
              <nav aria-label="About CoastWatch" className="flex flex-wrap gap-x-5 gap-y-2 sm:ml-auto">
                <Link href="/about" className="font-medium text-accent hover:underline">About</Link>
                <Link href="/sources" className="font-medium text-accent hover:underline">Data &amp; sources</Link>
                <a href={RESEARCH.doi} target="_blank" rel="noreferrer" className="font-medium text-accent hover:underline">Research paper ↗</a>
              </nav>
            </div>
          </footer>
        )}
        <TabBar active={active} />
      </div>
      <OfficialDrawer />
    </OfficialProvider>
    </MotionConfig>
  );
}
