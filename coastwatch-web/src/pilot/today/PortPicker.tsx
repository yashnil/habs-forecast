"use client";

import "./today.css";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { TopBar } from "@/v4/shell/TopBar";
import { PilotDock, PilotFooter } from "@/pilot/shell/PilotShell";
import { SkipLink } from "@/v4/shell/SkipLink";
import { readHome } from "@/v4/coast/home";
import { Icon } from "@/v4/ui/Icon";
import type { PortOption } from "./load";
import { readTodayPlace } from "./navPlace";

/**
 * /today with no port: the reader's last Today port or home port opens at once. Otherwise this is a
 * calm picker, pilot ports first, then every California port with a filter.
 */
export function PortPicker({ ports, requested, pick, now }: { ports: PortOption[]; requested: string | null; pick: boolean; now: string }) {
  const router = useRouter();
  const [q, setQ] = useState("");
  useEffect(() => {
    if (pick || requested) return;
    const known = new Set(ports.map((p) => p.place_id));
    const home = readHome()?.place;
    const target = [readTodayPlace(), home].find((p): p is string => !!p && known.has(p));
    if (target) router.replace(`/today?place=${target}`);
  }, [pick, requested, ports, router]);
  const pilot = ports.filter((p) => p.pilot);
  const rest = useMemo(() => {
    const s = q.trim().toLowerCase();
    return ports.filter((p) => !p.pilot && (!s || p.label.toLowerCase().includes(s) || (p.county ?? "").toLowerCase().includes(s)));
  }, [ports, q]);
  return (
    <div className="env-instrument today min-h-dvh pb-[calc(76px+env(safe-area-inset-bottom,0px))] lg:pb-10" data-testid="today-picker">
      <SkipLink />
      <TopBar active="today" now={new Date(now)} />
      <main id="main">
        <header className="today-hero">
          <div className="today-wrap pb-8 pt-6 md:pt-10">
            <p className="nx-caps">Today at your port</p>
            <h1 className="today-port mt-2 text-n-ink">Choose your port</h1>
            <p className="today-lede mt-3 text-n-ink-2">Official status for your fisheries, the latest toxin tests, sea and weather, the bloom outlook and whale rules, in one page.</p>
            {requested && (
              <p className="mt-3 text-[15px] t-amber" role="status" data-testid="today-unknown-port">
                Neritic does not know the port &ldquo;{requested}&rdquo;. Choose one below.
              </p>
            )}
          </div>
        </header>
        <div className="today-wrap">
          <section className="today-sec" aria-labelledby="pilot-h">
            <header>
              <h2 id="pilot-h" className="today-h2">
                Pilot ports
              </h2>
            </header>
            <ul className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {pilot.map((p) => (
                <li key={p.place_id}>
                  <Link href={`/today?place=${p.place_id}`} className="today-action nx-focus nx-press nx-tint nx-lift min-h-[76px]" data-testid={`pick-port-${p.place_id}`}>
                    <span className="font-serif text-[24px] leading-tight text-n-ink">{p.label}</span>
                    <small>{p.county ? `${p.county} County` : "California"}</small>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
          <section className="today-sec" aria-labelledby="all-h">
            <header>
              <h2 id="all-h" className="today-h2">
                All California ports
              </h2>
            </header>
            <label className="mt-3 flex min-h-12 items-center gap-2 rounded-[4px] border border-n-line-strong px-3 focus-within:border-n-ink-2">
              <Icon name="search" className="h-[18px] w-[18px] text-n-ink-3" />
              <span className="sr-only">Filter ports</span>
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter by port or county" className="min-h-11 w-full bg-transparent text-[16px] text-n-ink outline-none placeholder:text-n-ink-3" data-testid="port-filter" />
            </label>
            <ul className="mt-2 columns-1 gap-8 sm:columns-2 lg:columns-3">
              {rest.map((p) => (
                <li key={p.place_id} className="break-inside-avoid">
                  <Link href={`/today?place=${p.place_id}`} className="nx-focus flex min-h-11 items-center justify-between gap-3 border-b border-n-line text-[15.5px] text-n-ink hover:text-n-accent">
                    <span>{p.label}</span>
                    <span className="text-[13px] text-n-ink-3">{p.county ?? ""}</span>
                  </Link>
                </li>
              ))}
            </ul>
            {!rest.length && <p className="today-src mt-3">No port matches &ldquo;{q}&rdquo;.</p>}
          </section>
          <p className="nx-margin mt-8">The pilot ports have the most complete data; elsewhere some sections may be empty.</p>
          <PilotFooter />
        </div>
      </main>
      <PilotDock active="today" />
    </div>
  );
}
