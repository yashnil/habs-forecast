import Link from "next/link";
import { EXPERIENCES } from "@/content/copy";
import { DemoAboutButton, DemoIntro } from "@/components/DemoIntro";
import { SourceHealthDot } from "@/components/SourceHealthDot";
import { DEMO } from "@/lib/demo";
import { RegulatoryGapBanner } from "@/components/RegulatoryGap";
import { OfficialDrawer, OfficialPill, OfficialProvider, TabBar } from "@/components/shell/OfficialShell";
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
  active: "map" | "bloom" | "fisheries" | "sources";
  manifest: Manifest | null;
  official?: OfficialDataset | null;
  officialStatus?: SourceStatus | null;
  banner?: React.ReactNode;
}) {
  return (
    <OfficialProvider official={official} status={officialStatus}>
      <div className={`flex min-h-dvh flex-col bg-page pb-[var(--cw-tabbar-h)] md:pb-0 ${scroll === "app" ? "lg:h-dvh" : ""}`}>
        <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-navy-950 text-on-navy">
          <div className="flex h-[52px] items-center gap-6 px-4 md:h-[var(--cw-masthead-h)] md:gap-8 md:px-6">
            <Link href="/" className="flex items-center gap-2.5" aria-label="CoastWatch home">
              <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
                <rect width="32" height="32" rx="8" fill="#10263e" />
                <path d="M6 19c3.2-2.6 6.4-2.6 9.6 0s6.6 2.6 10.4-.2" fill="none" stroke="#6cc6dc" strokeWidth="2.2" strokeLinecap="round" />
                <path d="M6 13.5c3.2-2.6 6.4-2.6 9.6 0s6.6 2.6 10.4-.2" fill="none" stroke="#e8eef5" strokeOpacity=".55" strokeWidth="2.2" strokeLinecap="round" />
                <circle cx="22.5" cy="8.5" r="2.2" fill="#f6bb5c" />
              </svg>
              <span className="text-[17px] font-semibold tracking-tight text-white">CoastWatch</span>
              <span className="hidden border-l border-white/15 pl-2.5 text-[13px] text-on-navy-2 lg:inline">California</span>
            </Link>
            {DEMO ? (
              <span className="hidden rounded-full border border-white/20 px-2.5 sm:inline-block py-0.5 text-[12px] font-medium uppercase tracking-[0.08em] text-on-navy-2" data-testid="demo-badge">
                Research preview
              </span>
            ) : (
            <nav aria-label="Primary" className="hidden h-full md:flex">
              <ul className="flex h-full gap-1">
                {EXPERIENCES.map((e) => {
                  const current = active === e.key;
                  return (
                    <li key={e.key} className="h-full">
                      <Link
                        href={e.href}
                        aria-current={current ? "page" : undefined}
                        data-testid={`nav-${e.key}`}
                        className={`-mb-px flex h-full items-center whitespace-nowrap border-b-2 px-3 text-[15px] font-medium ${
                          current ? "border-accent-on-navy text-white" : "border-transparent text-on-navy-2 hover:text-on-navy"
                        }`}
                      >
                        {e.label}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </nav>
            )}
            <div className="ml-auto flex items-center gap-3">
              {DEMO && <DemoAboutButton className="hidden rounded-md px-1.5 py-1.5 text-[14px] text-on-navy-2 hover:text-on-navy md:inline" />}
              <OfficialPill />
              <Link
                href="/sources"
                data-testid="data-status-link"
                aria-current={active === "sources" ? "page" : undefined}
                className={`flex items-center gap-2 rounded-md px-1.5 py-1.5 text-[14px] ${active === "sources" ? "text-white" : "text-on-navy-2 hover:text-on-navy"}`}
              >
                <SourceHealthDot manifest={manifest} />
                <span className="hidden sm:inline">Data status</span>
                <span className="sr-only sm:hidden">Data status</span>
              </Link>
            </div>
          </div>
        </header>
        {DEMO && <RegulatoryGapBanner />}
        {banner}
        <main className={`flex min-h-0 flex-1 flex-col ${scroll === "app" ? "theme-dark" : ""}`}>{children}</main>
        <TabBar active={active} />
      </div>
      <OfficialDrawer />
      {DEMO && <DemoIntro />}
    </OfficialProvider>
  );
}
