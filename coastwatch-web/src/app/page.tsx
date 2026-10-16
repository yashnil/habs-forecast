import { loadData } from "@/lib/data";
import { Explore } from "@/v4/explore/Explore";
import { ReadinessProvider } from "@/v4/readiness/store";
import { DataUnavailable } from "@/v4/shell/DataUnavailable";

// Artifacts change several times a day; read the manifest per request (remote fetches are cached 5 min).
export const dynamic = "force-dynamic";

/** Explore: the map-first workspace (v4). v2 links (region, port, var, lead, layer, inspect) open the same view. */
export default async function Home({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const data = await loadData();
  const sp = await searchParams;
  const initialParams = Object.fromEntries(Object.entries(sp).filter(([, v]) => typeof v === "string")) as Record<string, string>;
  return (
    <ReadinessProvider>
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
