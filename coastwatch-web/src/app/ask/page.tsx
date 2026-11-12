import { askCoverage } from "@/pilot/ask/ui/coverage";
import { shellPorts } from "@/nx/shell/ports.server";
import { askDataset, meta } from "@/pilot/ask/service";
import { suggestionsFor } from "@/pilot/ask/suggestions";
import { AskChat } from "@/pilot/ask/ui/AskChat";
import { PilotShell } from "@/pilot/shell/PilotShell";
import { loadTrust } from "@/lib/data";
import { pilotAccountConfig } from "@/pilot/account/config";

export const dynamic = "force-dynamic";
export const metadata = { title: "Ask Neritic" };

/**
 * /ask?place=port-592 — Ask Neritic for one pilot port (private pilot), in the pilot's own shell (the
 * same top bar, dock and Almanac ground as Today). Readiness and the dataset label are decided on the
 * server from configuration and the manifest only. Today opens the same chat in a sheet.
 *
 * Ask answers only for the pilot ports. A link from any other port (`?place=port-473`) says that port is not
 * covered yet and waits for the reader to choose a pilot port: it never opens on another port in its place.
 */
export default async function AskPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const sp = await searchParams;
  const raw = typeof sp.place === "string" ? sp.place : null;
  const names = Object.fromEntries((await shellPorts()).map((p) => [p.key, p.name]));
  const coverage = askCoverage(raw, names);
  const place = coverage.state === "uncovered" ? "" : coverage.place;
  return (
    <PilotShell testid="ask-page" account={pilotAccountConfig()} trust={await loadTrust()}>
      <AskChat
        initialPlace={place}
        uncovered={coverage.state === "uncovered" ? coverage.port : null}
        portNames={names}
        preferSaved={coverage.state === "open"}
        meta={meta(process.env, await askDataset())}
        suggestions={place ? suggestionsFor(place) : []}
      />
    </PilotShell>
  );
}
