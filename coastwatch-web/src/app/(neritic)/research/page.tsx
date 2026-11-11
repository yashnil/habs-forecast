import { loadBloomData, loadData } from "@/lib/data";
import { Research } from "@/nx/research/Research";
import { ResearchUnavailable } from "@/nx/research/Unavailable";

export const metadata = { title: "Research" };

/** The clock: the server's, or the fixture's in tests (CW_FIXED_NOW), so every date on the page is computed once. */
const serverNow = () => (process.env.CW_FIXED_NOW ? new Date(process.env.CW_FIXED_NOW) : new Date()).toISOString();

/**
 * Research (journeys.md Journey 3): a station's record as strips on one time axis with one brush, its checks,
 * provenance, dated log and exports, beside the chart where its stations sample. The data are read on the
 * server, so the record is in the first HTML.
 */
export default async function ResearchPage() {
  const [data, bloom] = await Promise.all([loadData(), loadBloomData()]);
  if (!data.ok) return <ResearchUnavailable why={data.error} />;
  if (!bloom.ok || !bloom.observations) return <ResearchUnavailable why={bloom.ok ? (bloom.observationsError ?? "The shore-station samples are not in this dataset.") : bloom.error} />;
  return <Research observations={bloom.observations} ports={data.ports} manifest={data.manifest} official={data.official} baseUrl={data.baseUrl} now={serverNow()} />;
}
