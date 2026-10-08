import { AppShell } from "@/components/AppShell";
import { DataBanners } from "@/components/Banners";
import { LiveOceanMap } from "@/components/LiveOceanMap";
import { OfficialStatusCard } from "@/components/panels/OfficialStatusCard";
import { loadData } from "@/lib/data";

// Artifacts change several times a day; read the manifest per request (remote fetches are cached 5 min).
export const dynamic = "force-dynamic";

export default async function Home() {
  const data = await loadData();
  return (
    <AppShell active="map" manifest={data.ok ? data.manifest : null} banner={<DataBanners manifest={data.ok ? data.manifest : null} error={data.ok ? null : data.error} />}>
      {data.ok ? (
        <LiveOceanMap manifest={data.manifest} ports={data.ports} portsError={data.portsError} baseUrl={data.baseUrl} />
      ) : (
        <div className="mx-auto w-full max-w-xl p-4">
          <OfficialStatusCard />
        </div>
      )}
    </AppShell>
  );
}
