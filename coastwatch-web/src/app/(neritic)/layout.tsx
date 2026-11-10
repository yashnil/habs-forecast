import type { Metadata, Viewport } from "next";
import "@/nx/ds/tokens.css";
import "@/nx/chart/chart.css";
import "@/nx/shell/shell.css";
import { literata, publicSans } from "@/nx/ds/fonts";
import { PAPER_DAY } from "@/nx/ds/palette";
import { ChartHost } from "@/nx/chart/ChartHost";
import { OpeningController, OpeningStill } from "@/nx/opening/Opening";
import { ENTRY_SCRIPT } from "@/nx/shell/entry";
import { MarginFrame } from "@/nx/shell/Margin";
import { PortFinder } from "@/nx/shell/PortFinder";
import { PortsProvider } from "@/nx/shell/port";
import { Shell } from "@/nx/shell/Shell";
import { shellPorts } from "@/nx/shell/ports.server";

// the port list is read per request (the dataset is republished several times a day)
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: { default: "Neritic: season status, crab tests and the next date for your port", template: "%s · Neritic" },
};

export const viewport: Viewport = { themeColor: PAPER_DAY, colorScheme: "light dark" };

/**
 * The Neritic v12 route group: `/`, Today, Explore, My Coast and Research share one layout, so the chart (one
 * MapLibre instance), the bar and the port survive every navigation between them; only the margin changes.
 * URLs are unchanged by the group. Legacy pages (/about, /account, /ask, …) keep their own shell outside it.
 */
export default async function NeriticLayout({ children }: { children: React.ReactNode }) {
  const ports = await shellPorts();
  return (
    <div className={`n12-app ${literata.variable} ${publicSans.variable}`} data-testid="nx-app">
      {/* before the first paint: what `/` is (home, the opening, or Today at the saved port) and the chart mode */}
      <script dangerouslySetInnerHTML={{ __html: ENTRY_SCRIPT }} />
      <PortsProvider value={ports}>
        <Shell />
        <ChartHost ports={ports.map((p) => ({ key: p.key, name: p.name, lon: p.lon, lat: p.lat, pilot: p.pilot }))} opening={<OpeningStill />} />
        <MarginFrame>{children}</MarginFrame>
        <PortFinder />
        <OpeningController />
      </PortsProvider>
    </div>
  );
}
