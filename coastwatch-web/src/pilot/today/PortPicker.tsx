"use client";

import "./today.css";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import type { Map as MlMap } from "maplibre-gl";
import { cameraTier, journeyTo } from "@/lib/camera";
import { OceanStage } from "@/v4/stage/OceanStage";
import type { StageData, StageMark } from "@/v4/stage/types";
import { TopBar } from "@/v4/shell/TopBar";
import { useReportTrust, type Trust } from "@/v4/shell/Notices";
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
export function PortPicker({ ports, requested, pick, now, trust, stage }: { ports: PortOption[]; requested: string | null; pick: boolean; now: string; trust: Trust | null; stage: StageData | null }) {
  const router = useRouter();
  useReportTrust(trust);
  const [q, setQ] = useState("");
  useEffect(() => {
    if (pick || requested) return;
    const known = new Set(ports.map((p) => p.place_id));
    const home = readHome()?.place;
    const target = [readTodayPlace(), home].find((p): p is string => !!p && known.has(p));
    if (target) router.replace(`/today?place=${target}`);
  }, [pick, requested, ports, router]);
  const map = useRef<MlMap | null>(null);
  const pilot = ports.filter((p) => p.pilot);
  const rest = useMemo(() => {
    const s = q.trim().toLowerCase();
    return ports.filter((p) => !p.pilot && (!s || p.label.toLowerCase().includes(s) || (p.county ?? "").toLowerCase().includes(s)));
  }, [ports, q]);
  const marks: StageMark[] = pilot.map((p) => ({ id: p.place_id, lon: p.lon, lat: p.lat, kind: "port", label: p.label }));
  /** The descent: the camera flies to the port, then its Today opens (reduced motion and a stage without a map go straight there). */
  const fly = (e: MouseEvent<HTMLAnchorElement>, p: PortOption) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || !map.current || cameraTier() === "none") return;
    e.preventDefault();
    journeyTo(map.current, { center: [p.lon, p.lat], zoom: 9.3 });
    window.setTimeout(() => router.push(`/today?place=${p.place_id}`), 1300);
  };
  return (
    <div className="env-instrument today min-h-dvh pb-[calc(76px+env(safe-area-inset-bottom,0px))] lg:pb-0" data-testid="today-picker">
      <SkipLink />
      <TopBar active="today" now={new Date(now)} trust={trust} over />
      <main id="main">
        <div className="today-stack">
          <OceanStage className="today-stage today-stage-coast" center={[-120.6, 36.9]} zoom={5.6} phoneZoom={4.9} pitch={0} bearing={0} field="none" currents caption={stage?.currents ? { what: "Surface currents", kind: "observed", date: stage.currents.hour } : false} data={stage} marks={marks} arrival="coast" scrollLinked onMap={(m) => (map.current = m)}>
            <div className="today-over" data-testid="today-hero">
              <div data-beat="1">
                <h1 className="nx-display nx-display-hero nx-scroll-title today-port">Where do you fish?</h1>
              </div>
              <p className="nx-lede today-lede" data-beat="2">
                Official status for your fisheries, the latest toxin tests, the sea and the bloom outlook, on one page.
              </p>
              {requested && (
                <p className="today-cue today-old" role="status" data-testid="today-unknown-port">
                  Neritic does not know the port &ldquo;{requested}&rdquo;. Choose one below.
                </p>
              )}
              <ul className="today-glance today-pick" data-beat="3">
                {pilot.map((p) => (
                  <li key={p.place_id}>
                    <Link href={`/today?place=${p.place_id}`} onClick={(e) => fly(e, p)} className="today-gl nx-focus nx-press" data-testid={`pick-port-${p.place_id}`}>
                      <span className="today-gl-v today-pick-name">{p.label}</span>
                      <span className="today-gl-s">{p.county ? `${p.county} County` : "California"}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </OceanStage>
          <div className="today-pin" aria-hidden />
        </div>
        <div className="today-brief">
          <div className="today-brief-in">
            <section className="today-ch" aria-labelledby="all-h">
              <div className="today-ch-head">
                <h2 id="all-h" className="nx-title">
                  All California ports
                </h2>
                <p className="today-say">The pilot ports have the most complete data; elsewhere some sections may be empty.</p>
              </div>
              <div className="today-ch-body">
                <label className="flex min-h-12 items-center gap-2 rounded-[var(--nx-r-12)] bg-[var(--nx-surface-2)] px-3">
                  <Icon name="search" className="h-[18px] w-[18px] text-n-ink-3" />
                  <span className="sr-only">Filter ports</span>
                  <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter by port or county" className="min-h-11 w-full bg-transparent text-[16px] text-n-ink outline-none placeholder:text-n-ink-3" data-testid="port-filter" />
                </label>
                <ul className="mt-3 columns-1 gap-8 sm:columns-2">
                  {rest.map((p) => (
                    <li key={p.place_id} className="break-inside-avoid">
                      <Link href={`/today?place=${p.place_id}`} onClick={(e) => fly(e, p)} className="nx-focus flex min-h-11 items-center justify-between gap-3 border-b border-n-line text-[15.5px] text-n-ink hover:underline">
                        <span>{p.label}</span>
                        <span className="text-[13px] text-n-ink-3">{p.county ?? ""}</span>
                      </Link>
                    </li>
                  ))}
                </ul>
                {!rest.length && <p className="today-note mt-3">No port matches &ldquo;{q}&rdquo;.</p>}
              </div>
            </section>
            <PilotFooter />
          </div>
        </div>
      </main>
      <PilotDock active="today" />
    </div>
  );
}
