import type { Metadata, Viewport } from "next";
import "@/nx/ds/tokens.css";
import "@/nx/chart/chart.css";
import "@/nx/shell/shell.css";
import { PAPER_DAY } from "@/nx/ds/palette";
import { AlertWatch } from "@/nx/alerts/AlertWatch";
import { ChartHost } from "@/nx/chart/ChartHost";
import { MotionPrefs } from "@/nx/ds/MotionPrefs";
import { OpeningController, OpeningStill } from "@/nx/opening/Opening";
import { ENTRY_SCRIPT } from "@/nx/shell/entry";
import { Workspace } from "@/nx/shell/Margin";
import { PortFinder } from "@/nx/shell/PortFinder";
import { PortsProvider } from "@/nx/shell/port";
import { Shell } from "@/nx/shell/Shell";
import { shellPorts } from "@/nx/shell/ports.server";
import { finderData } from "@/nx/home/finder.server";

// the port list is read per request (the dataset is republished several times a day)
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  // absolute: the root layout's "— Neritic" template must not repeat the name after this one
  title: { absolute: "Neritic: season status, crab tests and the next date for your port", template: "%s · Neritic" },
};

export const viewport: Viewport = { themeColor: PAPER_DAY, colorScheme: "light dark" };

/**
 * The Neritic route group: `/`, Today, Explore, My Coast and Research share one layout, so the chart (one
 * MapLibre instance), the bar and the port survive every navigation between them; only the page changes. The
 * workspace lays the page and the chart pane out from the page's declared mode (chart or desk, v13 README §1).
 * URLs are unchanged by the group. Legacy pages (/about, /account, /ask, …) keep their own shell outside it.
 */
export default async function NeriticLayout({ children }: { children: React.ReactNode }) {
  const [ports, finder] = await Promise.all([shellPorts(), finderData()]);
  return (
    <div className="n12-app" data-testid="nx-app">
      {/* before the first paint: what `/` is (home, the opening, or Today at the saved port) and the chart mode */}
      <script dangerouslySetInnerHTML={{ __html: ENTRY_SCRIPT }} />
      <MotionPrefs />
      {/* the alert check for followed ports: draws nothing, runs after the page has drawn (src/nx/alerts) */}
      <AlertWatch />
      <PortsProvider value={ports}>
        <Shell />
        <Workspace chart={<ChartHost ports={ports.map((p) => ({ key: p.key, name: p.name, lon: p.lon, lat: p.lat, pilot: p.pilot }))} opening={<OpeningStill />} />}>{children}</Workspace>
        <PortFinder finder={finder} />
        <OpeningController />
      </PortsProvider>
    </div>
  );
}
