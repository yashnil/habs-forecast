import { Suspense } from "react";
import { redirect } from "next/navigation";
import { loadData } from "@/lib/data";
import { Briefing } from "@/v4/coast/Briefing";
import { ReadinessProvider } from "@/v4/readiness/store";
import { DataUnavailable } from "@/v4/shell/DataUnavailable";

export const dynamic = "force-dynamic";
export const metadata = { title: "My Coast" };

/** My Coast (v4): the answer-first trip briefing for one place. v2 links (place, for, species, date) are unchanged. */
export default async function CoastPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  // v2 links that asked for My Coast in Research mode open the Research workspace for that place
  const sp = await searchParams;
  if (sp.mode === "research") redirect(`/research${typeof sp.place === "string" ? `?place=${encodeURIComponent(sp.place)}` : ""}`);
  const data = await loadData();
  if (!data.ok) return <DataUnavailable error={data.error} />;
  return (
    <ReadinessProvider>
      <Suspense fallback={null}>
        <Briefing manifest={data.manifest} ports={data.ports} official={data.official} stations={data.stations} stationsError={data.stationsError} baseUrl={data.baseUrl} portIntel={data.portIntel} />
      </Suspense>
    </ReadinessProvider>
  );
}
