import Link from "next/link";
import { BRAND } from "@/content/brand";
import { DISCLAIMER, RESEARCH } from "@/content/copy";
import { loadData } from "@/lib/data";
import { charmRun } from "@/lib/layers";
import { InfoPage, InfoSection, infoStyles as s } from "@/nx/pages/InfoPage";
import { LiveSources } from "@/nx/pages/LiveSources";
import { OfficialSources } from "@/nx/pages/OfficialSources";
import { registryTrust } from "@/nx/pages/registry.server";
import { SourceRows } from "@/nx/pages/SourceRows";

export const dynamic = "force-dynamic";
export const metadata = { title: "Data & sources" };

const CONTENTS = [
  { id: "official-status-h", label: "Official closures and advisories" },
  { id: "research-h", label: "The research" },
  { id: "src-h", label: "Source status" },
  { id: "live-h", label: "Live sources" },
  { id: "freshness-h", label: "How freshness is decided" },
];

/**
 * Data & sources (v15.1: a page of the shell, src/nx/pages): what Neritic is not, where official notices come from,
 * the research, every source the dataset reads with its status, and how freshness is decided. Words unchanged.
 */
export default async function SourcesPage() {
  const [data, trust] = await Promise.all([loadData(), registryTrust()]);
  const manifest = data.ok ? data.manifest : null;
  return (
    <InfoPage
      id="sources"
      kicker="Provenance"
      title={<>Data &amp; sources</>}
      lede={DISCLAIMER}
      trust={trust}
      contents={CONTENTS}
      head={
        <p className={s.small} data-testid="sources-independence">
          {BRAND.independence} Satellite data credited to NOAA CoastWatch below come from that NOAA program.
        </p>
      }
    >
      <div className={s.section}>
        <OfficialSources />
      </div>
      <InfoSection id="research-h" title="The research behind Neritic" data-testid="research">
        <div className={s.prose}>
          <p>
            <strong>{RESEARCH.title}</strong>
            <br />
            {RESEARCH.author} · {RESEARCH.venue} · {RESEARCH.published} ·{" "}
            <a href={RESEARCH.doi} target="_blank" rel="noreferrer">
              doi:10.33422/ccgconf.v2i2.1619
            </a>
          </p>
          <p>{RESEARCH.relation}</p>
          <p>
            <Link href="/about">About Neritic: what it shows, what it cannot tell you, and the research</Link>
          </p>
        </div>
      </InfoSection>
      <div className={s.section}>
        {manifest ? (
          <SourceRows
            sources={manifest.sources}
            run={charmRun(manifest)}
            pipeline={{ runId: manifest.pipeline_run_id, version: manifest.pipeline_version, generatedAt: manifest.generated_at }}
            registry={trust.state}
          />
        ) : (
          <p className={s.note}>Source status cannot be shown because the data manifest is unavailable.</p>
        )}
      </div>
      <div className={s.section}>
        <LiveSources />
      </div>
      <InfoSection id="freshness-h" title="How freshness is decided">
        <div className={s.prose}>
          <p>
            Each source publishes its own dates. Your browser compares them with today&apos;s date: <strong>Current</strong> means within the source&apos;s normal update window, <strong>Stale</strong> means newer data should exist but has not been published or
            fetched, and <strong>Historical</strong> means the data is too old to describe current conditions. If an update fails, the last successful data stays on screen with its real dates and a failure notice.
          </p>
        </div>
      </InfoSection>
    </InfoPage>
  );
}
