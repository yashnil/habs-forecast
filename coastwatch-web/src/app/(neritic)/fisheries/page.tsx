import { FisheriesExposure } from "@/components/fisheries/FisheriesExposure";
import { FisheriesByMode } from "@/components/fisheries/FisheriesByMode";
import { FISHERIES_COPY } from "@/content/copy";
import { loadFisheriesData } from "@/lib/data";
import { parseView } from "@/lib/fisheries";
import { sourceStatus } from "@/lib/layers";
import { InfoPage, infoStyles as s } from "@/nx/pages/InfoPage";
import { FisheriesFrame } from "@/nx/pages/FisheriesFrame";
import { OfficialSources } from "@/nx/pages/OfficialSources";
import { registryTrust } from "@/nx/pages/registry.server";
import "@/nx/pages/bridge.css";

export const dynamic = "force-dynamic";
export const metadata = { title: "Fisheries & Economic Exposure" };

const str = (v: string | string[] | undefined) => (typeof v === "string" ? v : undefined);

/**
 * Fisheries & Economic Exposure (v15.1: a page of the shell, src/nx/pages). The story, chart, table and methods are
 * the M7 components, drawn in the design system's colours and type through the token bridge (src/nx/pages/bridge.css);
 * their words and numbers are unchanged.
 */
export default async function FisheriesPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const [data, trust, sp] = await Promise.all([loadFisheriesData(), registryTrust(), searchParams]);
  if (data.ok && data.fisheries) {
    return (
      <InfoPage id="fisheries" trust={trust} width="wide">
        <FisheriesFrame official={data.official} status={sourceStatus(data.manifest, "official")}>
          <FisheriesByMode official={data.official}>
            <FisheriesExposure
              ds={data.fisheries}
              status={sourceStatus(data.manifest, "foss_landings")}
              official={data.official}
              initial={parseView(data.fisheries, { unit: str(sp.unit), tiers: str(sp.tiers), group: str(sp.group), year: str(sp.year) })}
            />
          </FisheriesByMode>
        </FisheriesFrame>
      </InfoPage>
    );
  }
  return (
    <InfoPage id="fisheries" testId="fisheries-unavailable" kicker="Historical context · Statewide" title={FISHERIES_COPY.heading} lede={FISHERIES_COPY.definition} trust={trust}>
      <p role="alert" className={`${s.note} ${s.noteReg}`}>
        <strong>{FISHERIES_COPY.unavailable}</strong> {data.ok ? (data.fisheriesError === "not published" ? "This dataset was published before fisheries data were added." : data.fisheriesError) : data.error} No landings are shown.
      </p>
      <div className={s.section}>
        <OfficialSources />
      </div>
    </InfoPage>
  );
}
