import { AppShell } from "@/components/AppShell";
import { DataBanners } from "@/components/Banners";
import { FisheriesExposure } from "@/components/fisheries/FisheriesExposure";
import { Notice } from "@/components/ui/Primitives";
import { FISHERIES_COPY } from "@/content/copy";
import { loadFisheriesData } from "@/lib/data";
import { sourceStatus } from "@/lib/layers";

export const dynamic = "force-dynamic";
export const metadata = { title: "Fisheries & Economic Exposure — CoastWatch" };

export default async function FisheriesPage() {
  const data = await loadFisheriesData();
  const manifest = data.ok ? data.manifest : null;
  return (
    <AppShell active="fisheries" manifest={manifest} banner={<DataBanners manifest={manifest} error={data.ok ? null : data.error} />} scroll="page">
      {data.ok && data.fisheries ? (
        <FisheriesExposure ds={data.fisheries} status={sourceStatus(data.manifest, "foss_landings")} official={data.official} />
      ) : (
        <div className="mx-auto w-full max-w-2xl space-y-4 px-4 py-6" data-testid="fisheries-unavailable">
          <h1 className="text-[22px] font-semibold tracking-tight text-ink">{FISHERIES_COPY.heading}</h1>
          <p className="text-[13px] text-ink-2">{FISHERIES_COPY.definition}</p>
          <Notice tone="warning" title={FISHERIES_COPY.unavailable}>
            {data.ok ? (data.fisheriesError === "not published" ? "This dataset was published before fisheries data were added." : data.fisheriesError) : data.error}
          </Notice>
        </div>
      )}
    </AppShell>
  );
}
