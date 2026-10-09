import { AppShell } from "@/components/AppShell";
import { DataBanners } from "@/components/Banners";
import { BloomIntelligence } from "@/components/bloom/BloomIntelligence";
import { OfficialStatusCard } from "@/components/panels/OfficialStatusCard";
import { Notice } from "@/components/ui/Primitives";
import { BLOOM_COPY } from "@/content/copy";
import { loadBloomData } from "@/lib/data";
import { sourceStatus } from "@/lib/layers";
import { DEFAULT_STATION } from "@/lib/observations";

export const dynamic = "force-dynamic";
export const metadata = { title: "Bloom Intelligence — CoastWatch" };

export default async function BloomPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const data = await loadBloomData();
  const sp = await searchParams;
  const manifest = data.ok ? data.manifest : null;
  const banner = <DataBanners manifest={manifest} error={data.ok ? null : data.error} />;
  if (!data.ok || !data.observations) {
    return (
      <AppShell active="bloom" manifest={manifest} official={data.ok ? data.official : null} officialStatus={data.ok ? sourceStatus(data.manifest, "official") : null} banner={banner} scroll="page">
        <div className="mx-auto w-full max-w-2xl space-y-4 px-4 py-6" data-testid="bloom-unavailable">
          <h1 className="text-[22px] font-semibold tracking-tight text-ink">{BLOOM_COPY.heading}</h1>
          <Notice tone="warning" title={BLOOM_COPY.unavailable}>
            {data.ok ? (data.observationsError === "not published" ? "This dataset was published before measured observations were added." : data.observationsError) : data.error} No
            measurements are shown, and this does not mean toxin is absent.
          </Notice>
          <OfficialStatusCard />
        </div>
      </AppShell>
    );
  }
  const obs = data.observations;
  const wanted = typeof sp.station === "string" ? sp.station : DEFAULT_STATION;
  const usable = obs.stations.filter((s) => s.status !== "failed");
  const station = usable.find((s) => s.station_id === wanted) ?? usable.find((s) => s.station_id === DEFAULT_STATION) ?? usable[0] ?? obs.stations[0];
  // the list and map need summaries only; the selected station carries its full series
  const lite = obs.stations.map((s) => ({ ...s, series: [], depths_m: [], sample_times: s.sample_times.slice(-1), qc: [], charm: null }));
  const relations = data.portIntel?.ports.find((p) => p.port_code === station.nearest_port_code)?.official_relations ?? [];
  const { variables, method, caveats, provenance, generated_at, window_start, program } = obs;
  return (
    <AppShell active="bloom" manifest={manifest} official={data.ok ? data.official : null} officialStatus={data.ok ? sourceStatus(data.manifest, "official") : null} banner={banner} scroll="page">
      <BloomIntelligence
          stations={lite}
          station={station}
          dataset={{ variables, method, caveats, provenance, generated_at, window_start, program }}
          status={sourceStatus(data.manifest, "calhabmap")}
          official={data.official}
          officialStatus={sourceStatus(data.manifest, "official")}
          officialError={data.officialError}
          relations={relations}
        />
    </AppShell>
  );
}
