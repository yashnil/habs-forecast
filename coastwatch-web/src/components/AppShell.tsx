import Link from "next/link";
import { EXPERIENCES } from "@/content/copy";
import { SourceHealthDot } from "@/components/SourceHealthDot";
import type { Manifest } from "@/generated/schema";

export function AppShell({
  children,
  active,
  manifest,
  banner,
}: {
  children: React.ReactNode;
  active: "map" | "sources";
  manifest: Manifest | null;
  banner?: React.ReactNode;
}) {
  return (
    <div className="flex min-h-dvh flex-col bg-page lg:h-dvh">
      <header className="z-20 border-b border-hairline bg-page/95 backdrop-blur">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5">
          <Link href="/" className="flex items-center gap-2" aria-label="CoastWatch home">
            <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
              <path d="M3 15c2.2 0 2.2-2 4.5-2s2.3 2 4.5 2 2.3-2 4.5-2 2.3 2 4.5 2" fill="none" stroke="var(--cw-accent)" strokeWidth="1.8" strokeLinecap="round" />
              <path d="M3 19c2.2 0 2.2-2 4.5-2s2.3 2 4.5 2 2.3-2 4.5-2 2.3 2 4.5 2" fill="none" stroke="var(--cw-ink-3)" strokeWidth="1.8" strokeLinecap="round" />
              <circle cx="12" cy="7" r="3" fill="var(--cw-forecast)" />
            </svg>
            <span className="text-[15px] font-semibold tracking-tight text-ink">CoastWatch</span>
            <span className="hidden text-[11px] text-ink-3 sm:inline">California</span>
          </Link>
          <nav aria-label="Experiences" className="order-3 w-full sm:order-none sm:w-auto">
            <ul className="flex gap-1 text-[12.5px]">
              {EXPERIENCES.map((e) => (
                <li key={e.key}>
                  {e.available && e.href ? (
                    <Link
                      href={e.href}
                      aria-current={active === "map" && e.key === "map" ? "page" : undefined}
                      className={`block whitespace-nowrap rounded-md px-2.5 py-1.5 font-medium ${
                        active === "map" && e.key === "map" ? "bg-surface-2 text-ink" : "text-ink-2 hover:text-ink"
                      }`}
                    >
                      {e.label}
                    </Link>
                  ) : (
                    <span
                      aria-disabled="true"
                      data-testid={`nav-upcoming-${e.key}`}
                      className="hidden cursor-default items-center gap-1.5 whitespace-nowrap rounded-md px-2.5 py-1.5 text-ink-3 md:flex"
                    >
                      {e.label}
                      <span className="rounded border border-hairline px-1 text-[9.5px] uppercase tracking-wider">Upcoming</span>
                    </span>
                  )}
                </li>
              ))}
              <li className="md:hidden">
                <span className="flex items-center whitespace-nowrap rounded-md px-2.5 py-1.5 text-[11.5px] text-ink-3" title="Bloom Intelligence, Fisheries & Economic Exposure, My Coast">
                  + 3 upcoming
                </span>
              </li>
            </ul>
          </nav>
          <div className="ml-auto flex items-center gap-3">
            <Link
              href="/sources"
              aria-current={active === "sources" ? "page" : undefined}
              className={`flex items-center gap-2 rounded-md px-2.5 py-1.5 text-[12.5px] font-medium ${active === "sources" ? "bg-surface-2 text-ink" : "text-ink-2 hover:text-ink"}`}
            >
              <SourceHealthDot manifest={manifest} />
              Data & sources
            </Link>
          </div>
        </div>
      </header>
      {banner}
      <main className="flex min-h-0 flex-1 flex-col">{children}</main>
    </div>
  );
}
