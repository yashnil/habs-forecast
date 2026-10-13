import { AppShell } from "@/components/AppShell";
import { DataBanners } from "@/components/Banners";
import { FisheriesExposure } from "@/components/fisheries/FisheriesExposure";
import { FisheriesByMode } from "@/components/fisheries/FisheriesByMode";
import { OfficialStatusCard } from "@/components/panels/OfficialStatusCard";
import { Notice } from "@/components/ui/Primitives";
import { FISHERIES_COPY } from "@/content/copy";
import { loadFisheriesData } from "@/lib/data";
import { parseView } from "@/lib/fisheries";
import { sourceStatus } from "@/lib/layers";

export const dynamic = "force-dynamic";
export const metadata = { title: "Fisheries & Economic Exposure — CoastWatch" };

const str = (v: string | string[] | undefined) => (typeof v === "string" ? v : undefined);

export default async function FisheriesPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const data = await loadFisheriesData();
  const sp = await searchParams;
  const manifest = data.ok ? data.manifest : null;
  return (
    <AppShell
      active="fisheries"
      manifest={manifest}
      official={data.ok ? data.official : null}
      officialStatus={data.ok ? sourceStatus(data.manifest, "official") : null}
      banner={<DataBanners manifest={manifest} error={data.ok ? null : data.error} />}
      scroll="page"
    >
      {data.ok && data.fisheries ? (
        <FisheriesByMode official={data.official}>
          <FisheriesExposure
            ds={data.fisheries}
            status={sourceStatus(data.manifest, "foss_landings")}
            official={data.official}
            initial={parseView(data.fisheries, { unit: str(sp.unit), tiers: str(sp.tiers), group: str(sp.group), year: str(sp.year) })}
          />
        </FisheriesByMode>
      ) : (
        <div className="mx-auto w-full max-w-2xl space-y-4 px-4 py-8" data-testid="fisheries-unavailable">
          <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-history">Historical context · Statewide</p>
          <h1 className="font-display text-[34px] font-medium leading-tight text-ink">{FISHERIES_COPY.heading}</h1>
          <p className="text-[15px] leading-relaxed text-ink-2">{FISHERIES_COPY.definition}</p>
          <Notice tone="warning" title={FISHERIES_COPY.unavailable}>
            {data.ok ? (data.fisheriesError === "not published" ? "This dataset was published before fisheries data were added." : data.fisheriesError) : data.error} No landings are shown.
          </Notice>
          <OfficialStatusCard />
        </div>
      )}
    </AppShell>
  );
}
