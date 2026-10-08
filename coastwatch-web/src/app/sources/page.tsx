import { AppShell } from "@/components/AppShell";
import { DataBanners } from "@/components/Banners";
import { OfficialStatusCard } from "@/components/panels/OfficialStatusCard";
import { SourceTable } from "@/components/SourceTable";
import { DISCLAIMER } from "@/content/copy";
import { loadData } from "@/lib/data";

export const dynamic = "force-dynamic";
export const metadata = { title: "Data & sources — CoastWatch" };

export default async function SourcesPage() {
  const data = await loadData();
  const manifest = data.ok ? data.manifest : null;
  return (
    <AppShell active="sources" manifest={manifest} banner={<DataBanners manifest={manifest} error={data.ok ? null : data.error} />}>
      <div className="mx-auto w-full max-w-4xl space-y-6 overflow-y-auto px-4 py-6">
        <header className="space-y-2">
          <h1 className="text-2xl font-semibold tracking-tight text-ink">Data & sources</h1>
          <p className="max-w-2xl text-[14px] leading-relaxed text-ink-2">{DISCLAIMER}</p>
        </header>
        <OfficialStatusCard />
        {manifest ? (
          <SourceTable manifest={manifest} official={data.ok ? data.official : null} />
        ) : (
          <p className="text-ink-2">Source status cannot be shown because the data manifest is unavailable.</p>
        )}
        <section className="space-y-2 text-[13px] leading-relaxed text-ink-2">
          <h2 className="text-[15px] font-semibold text-ink">How freshness is decided</h2>
          <p>
            Each source publishes its own dates. Your browser compares them with today&apos;s date: <strong className="text-ink">Current</strong>{" "}
            means within the source&apos;s normal update window, <strong className="text-ink">Stale</strong> means newer data should exist but has
            not been published or fetched, and <strong className="text-ink">Historical</strong> means the data is too old to describe current
            conditions. If an update fails, the last successful data stays on screen with its real dates and a failure notice.
          </p>
        </section>
      </div>
    </AppShell>
  );
}
