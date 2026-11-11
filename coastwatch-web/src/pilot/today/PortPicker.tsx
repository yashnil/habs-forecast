"use client";

import "./today.css";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, type MouseEvent } from "react";
import type { Map as MlMap } from "maplibre-gl";
import { cameraTier, journeyTo } from "@/lib/camera";
import { OceanStage } from "@/v4/stage/OceanStage";
import type { StageData, StageMark } from "@/v4/stage/types";
import { TopBar } from "@/v4/shell/TopBar";
import { useReportTrust, type Trust } from "@/v4/shell/Notices";
import { PilotDock, PilotFooter } from "@/pilot/shell/PilotShell";
import { SkipLink } from "@/v4/shell/SkipLink";
import { readHome } from "@/v4/coast/home";
import type { PortOption } from "./load";
import { PortDirectory } from "./PortDirectory";
import { readTodayPlace } from "./navPlace";

/**
 * /today with no port: the reader's last Today port or home port opens at once. Otherwise this is a
 * calm hero: the map, the pilot ports, and every other California port one quiet action away (PortDirectory).
 */
export function PortPicker({ ports, requested, pick, now, trust, stage }: { ports: PortOption[]; requested: string | null; pick: boolean; now: string; trust: Trust | null; stage: StageData | null }) {
  const router = useRouter();
  useReportTrust(trust);
  useEffect(() => {
    if (pick || requested) return;
    const known = new Set(ports.map((p) => p.place_id));
    const home = readHome()?.place;
    const target = [readTodayPlace(), home].find((p): p is string => !!p && known.has(p));
    if (target) router.replace(`/today?place=${target}`);
  }, [pick, requested, ports, router]);
  const map = useRef<MlMap | null>(null);
  const pilot = ports.filter((p) => p.pilot);
  const marks: StageMark[] = pilot.map((p) => ({ id: p.place_id, lon: p.lon, lat: p.lat, kind: "port", label: p.label }));
  /** The descent: the camera flies to the port, then its Today opens (reduced motion and a stage without a map go straight there). */
  const fly = (e: MouseEvent<HTMLAnchorElement>, p: PortOption) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || !map.current || cameraTier() === "none") return;
    e.preventDefault();
    journeyTo(map.current, { center: [p.lon, p.lat], zoom: 9.3 });
    // the flight is a courtesy, never a wait: a device that cannot fly it smoothly goes straight on
    window.setTimeout(() => router.push(`/today?place=${p.place_id}`), cameraTier() === "full" ? 1300 : 300);
  };
  return (
    <div className="env-instrument today today-front min-h-dvh pb-[calc(76px+env(safe-area-inset-bottom,0px))] lg:pb-0" data-testid="today-picker">
      <SkipLink />
      <TopBar active="today" now={new Date(now)} trust={trust} over />
      <main id="main">
        <div className="today-stack">
          <OceanStage className="today-stage today-stage-coast" center={[-120.6, 36.9]} zoom={5.6} phoneZoom={4.9} pitch={0} bearing={0} field="none" currents caption={stage?.currents ? { what: "Surface currents", kind: "observed", date: stage.currents.hour } : false} data={stage} marks={marks} arrival="coast" onMap={(m) => (map.current = m)}>
            <div className="today-over" data-testid="today-hero">
              <div data-beat="1">
                <h1 className="nx-display nx-display-hero today-port">Where do you fish?</h1>
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
              <div data-beat="3">
                <PortDirectory ports={ports} onFollow={fly} onOpenPort={(p) => router.push(`/today?place=${p.place_id}`)} />
              </div>
            </div>
          </OceanStage>
        </div>
        <div className="today-front-foot">
          <PilotFooter />
        </div>
      </main>
      <PilotDock active="today" />
    </div>
  );
}
