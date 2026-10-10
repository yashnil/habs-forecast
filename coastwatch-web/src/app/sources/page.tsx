import { AppShell } from "@/components/AppShell";
import { DataBanners } from "@/components/Banners";
import { OfficialStatusCard } from "@/components/panels/OfficialStatusCard";
import { SourceTable } from "@/components/SourceTable";
import { DISCLAIMER, RESEARCH } from "@/content/copy";
import { loadData } from "@/lib/data";
import { sourceStatus } from "@/lib/layers";

export const dynamic = "force-dynamic";
export const metadata = { title: "Data & sources — CoastWatch" };

export default async function SourcesPage() {
  const data = await loadData();
  const manifest = data.ok ? data.manifest : null;
  return (
    <AppShell
      active="sources"
      manifest={manifest}
      official={data.ok ? data.official : null}
      officialStatus={data.ok ? sourceStatus(data.manifest, "official") : null}
      banner={<DataBanners manifest={manifest} error={data.ok ? null : data.error} />} scroll="page">
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
        <section aria-labelledby="research-h" data-testid="research" className="space-y-2 border-t border-hairline pt-5 text-[13px] leading-relaxed text-ink-2">
          <h2 id="research-h" className="text-[15px] font-semibold text-ink">Research behind CoastWatch</h2>
          <p className="max-w-2xl">
            <span className="font-display text-[17px] leading-snug text-ink">{RESEARCH.title}</span>
            <br />
            {RESEARCH.author} · {RESEARCH.venue} · {RESEARCH.published} ·{" "}
            <a href={RESEARCH.doi} target="_blank" rel="noreferrer" className="font-medium text-accent underline underline-offset-2">
              doi:10.33422/ccgconf.v2i2.1619
            </a>
          </p>
          <p className="max-w-2xl">{RESEARCH.relation}</p>
        </section>
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
