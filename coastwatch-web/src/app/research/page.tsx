import { loadBloomData, loadData } from "@/lib/data";
import { Research } from "@/v4/research/Research";
import { ReadinessProvider } from "@/v4/readiness/store";
import { DataUnavailable } from "@/v4/shell/DataUnavailable";

export const dynamic = "force-dynamic";
export const metadata = { title: "Research" };

/** Research (v4): station records, the C-HARM history beside them, provenance, QC, exports and the native grid. */
export default async function ResearchPage() {
  const [data, bloom] = await Promise.all([loadData(), loadBloomData()]);
  if (!data.ok) return <DataUnavailable error={data.error} />;
  if (!bloom.ok || !bloom.observations) return <DataUnavailable error={bloom.ok ? (bloom.observationsError ?? "The observations are not published") : bloom.error} />;
  return (
    <ReadinessProvider>
      {/* no Suspense boundary: the page is dynamic, so the search params are read on the server and the
          content is in the first HTML (a streamed boundary is revealed only after React's ~300 ms batching) */}
      <Research manifest={data.manifest} ports={data.ports} observations={bloom.observations} official={data.official} baseUrl={data.baseUrl} />
    </ReadinessProvider>
  );
}
