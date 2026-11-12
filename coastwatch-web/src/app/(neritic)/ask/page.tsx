import { askCoverage } from "@/pilot/ask/ui/coverage";
import { shellPorts } from "@/nx/shell/ports.server";
import { askDataset, meta } from "@/pilot/ask/service";
import { suggestionsFor } from "@/pilot/ask/suggestions";
import { AskPage } from "@/nx/ask/AskPage";

export const dynamic = "force-dynamic";
export const metadata = { title: "Ask Neritic" };

/**
 * /ask?place=port-592: Ask Neritic for one pilot port, in the v13 shell (src/nx/ask). Readiness and the
 * dataset label are decided on the server from configuration and the manifest only.
 *
 * Ask answers only for the pilot ports. A link from any other port (`?place=port-473`) says that port is not
 * covered yet and waits for the reader to choose a pilot port: it never opens on another port in its place.
 */
export default async function Ask({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const sp = await searchParams;
  const raw = typeof sp.place === "string" ? sp.place : null;
  const names = Object.fromEntries((await shellPorts()).map((p) => [p.key, p.name]));
  const coverage = askCoverage(raw, names);
  const place = coverage.state === "uncovered" ? "" : coverage.place;
  return (
    <AskPage
      initialPlace={place}
      uncovered={coverage.state === "uncovered" ? coverage.port : null}
      preferSaved={coverage.state === "open"}
      meta={meta(process.env, await askDataset())}
      suggestions={place ? suggestionsFor(place) : []}
    />
  );
}
