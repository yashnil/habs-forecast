import { PILOT_PORTS } from "@/pilot/contracts";
import { portById } from "@/pilot/ask/places";
import { askDataset, meta } from "@/pilot/ask/service";
import { suggestionsFor } from "@/pilot/ask/suggestions";
import { AskChat } from "@/pilot/ask/ui/AskChat";
import { PilotShell } from "@/pilot/shell/PilotShell";
import { pilotAccountConfig } from "@/pilot/account/config";

export const dynamic = "force-dynamic";
export const metadata = { title: "Ask Neritic" };

/**
 * /ask?place=port-592 — Ask Neritic for one pilot port (private pilot), in the pilot's own shell (the
 * same top bar, dock and Almanac ground as Today). Readiness and the dataset label are decided on the
 * server from configuration and the manifest only. Today opens the same chat in a sheet.
 */
export default async function AskPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const sp = await searchParams;
  const raw = typeof sp.place === "string" ? sp.place : null;
  const place = portById(raw)?.place_id ?? PILOT_PORTS[0].place_id;
  return (
    <PilotShell testid="ask-page" account={pilotAccountConfig()}>
      <AskChat initialPlace={place} preferSaved={!raw} meta={meta(process.env, await askDataset())} suggestions={suggestionsFor(place)} />
    </PilotShell>
  );
}
