import { redirect } from "next/navigation";
import { loadData } from "@/lib/data";
import { frontDoorRedirect } from "@/pilot/frontDoor";
import { Explore } from "@/v4/explore/Explore";
import { ReadinessProvider } from "@/v4/readiness/store";
import { DataUnavailable } from "@/v4/shell/DataUnavailable";
import { Splash } from "@/v4/shell/Splash";

// Artifacts change several times a day; read the manifest per request (remote fetches are cached 5 min).
export const dynamic = "force-dynamic";

/**
 * Explore: the map-first workspace (v4). v2 links (region, port, var, lead, layer, inspect) open the same view.
 * Pilot (NERITIC_PILOT=1): a bare `/` opens Today at your port instead; any link that names what to
 * open still opens Explore, and `/?view=map` is Explore's own address (src/pilot/frontDoor.ts).
 */
export default async function Home({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const sp = await searchParams;
  const door = frontDoorRedirect(sp);
  if (door) redirect(door);
  const data = await loadData();
  const initialParams = Object.fromEntries(Object.entries(sp).filter(([, v]) => typeof v === "string")) as Record<string, string>;
  return (
    <ReadinessProvider>
      {/* the first visit's launch screen (the decision itself runs on every page, in the root layout) */}
      <Splash />
      <main id="main">
        {data.ok ? (
          <Explore
            manifest={data.manifest}
            ports={data.ports}
            official={data.official}
            portIntel={data.portIntel}
            stations={data.stations}
            stationsError={data.stationsError}
            baseUrl={data.baseUrl}
            initialParams={initialParams}
          />
        ) : (
          <DataUnavailable error={data.error} />
        )}
      </main>
    </ReadinessProvider>
  );
}
