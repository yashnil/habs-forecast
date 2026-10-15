import { AppShell } from "@/components/AppShell";
import { DataBanners } from "@/components/Banners";
import { MyCoast } from "@/components/coast/MyCoast";
import { OfficialStatusCard } from "@/components/panels/OfficialStatusCard";
import { loadData } from "@/lib/data";
import { sourceStatus } from "@/lib/layers";

export const dynamic = "force-dynamic";
export const metadata = { title: "My Coast" };

/** My Coast: one location's official information, measurements, forecast availability and observations. */
export default async function CoastPage() {
  const data = await loadData();
  return (
    <AppShell
      active="coast"
      manifest={data.ok ? data.manifest : null}
      official={data.ok ? data.official : null}
      officialStatus={data.ok ? sourceStatus(data.manifest, "official") : null}
      banner={<DataBanners manifest={data.ok ? data.manifest : null} error={data.ok ? null : data.error} />}
      scroll="page"
    >
      {data.ok ? (
        <MyCoast manifest={data.manifest} ports={data.ports} official={data.official} stations={data.stations} stationsError={data.stationsError} baseUrl={data.baseUrl} portIntel={data.portIntel} />
      ) : (
        <div className="mx-auto w-full max-w-xl p-4">
          <OfficialStatusCard />
        </div>
      )}
    </AppShell>
  );
}
