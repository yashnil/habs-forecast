import Link from "next/link";
import { PILOT_PORTS } from "@/pilot/contracts";
import { portById } from "@/pilot/ask/places";
import { askDataset, meta } from "@/pilot/ask/service";
import { suggestionsFor } from "@/pilot/ask/suggestions";
import { AskChat } from "@/pilot/ask/ui/AskChat";

export const dynamic = "force-dynamic";
export const metadata = { title: "Ask Neritic" };

/**
 * /ask?place=port-592 — Ask Neritic for one pilot port (private pilot). Standalone page; the
 * integrator may wrap it in the pilot shell. Readiness is decided on the server from config only.
 */
export default async function AskPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const sp = await searchParams;
  const raw = typeof sp.place === "string" ? sp.place : null;
  const place = portById(raw)?.place_id ?? PILOT_PORTS[0].place_id;
  return (
    <main className="min-h-dvh bg-n-bg font-[family-name:var(--nx-font-ui)] text-n-ink" data-testid="ask-page">
      <div className="mx-auto w-full max-w-[1120px] px-4 pb-16 pt-6 sm:px-8 sm:pt-10">
        <nav aria-label="Pilot" className="mb-6 flex flex-wrap gap-x-5 gap-y-2 text-[14px]">
          <Link href={`/today?place=${place}`} className="text-n-accent underline underline-offset-2">
            ← Today at {portById(place)?.label}
          </Link>
          <Link href="/" className="text-n-ink-2 underline underline-offset-2">
            Explore
          </Link>
        </nav>
        <AskChat initialPlace={place} meta={meta(process.env, await askDataset())} suggestions={suggestionsFor(place)} />
      </div>
    </main>
  );
}
