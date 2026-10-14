import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import { DataBanners } from "@/components/Banners";
import { SinceLastCheck } from "@/components/coast/SinceLastCheck";
import { speciesById } from "@/content/species";
import { coastHref, parseAudience, resolvePlace } from "@/lib/coast";
import { activityLabel } from "@/lib/keptBriefing";
import { loadData } from "@/lib/data";
import { sourceStatus } from "@/lib/layers";

export const dynamic = "force-dynamic";
export const metadata = { title: "What changed near your waters — CoastWatch" };

/**
 * Isolated prototype of "Since you last checked" (fishermen-first workstream). In the product it is
 * mounted on My Coast under the answer line (docs/coastwatch/fishermen-first/06-integration-contract.md);
 * this page exists so it can be tested and reviewed without editing My Coast.
 */
export default async function ChangesPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const q = await searchParams;
  const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? null;
  const data = await loadData();
  const place = data.ok ? resolvePlace(one(q.place), data.ports) : null;
  const audience = parseAudience(one(q.for));
  const species = speciesById(one(q.species));
  return (
    <AppShell
      active="coast"
      manifest={data.ok ? data.manifest : null}
      official={data.ok ? data.official : null}
      officialStatus={data.ok ? sourceStatus(data.manifest, "official") : null}
      banner={<DataBanners manifest={data.ok ? data.manifest : null} error={data.ok ? null : data.error} />}
      scroll="page"
    >
      <article className="mx-auto w-full max-w-[720px] px-4 pb-20 pt-6" data-testid="changes-page">
        <p className="text-[14px] text-ink-2">What changed near your waters · prototype</p>
        <h1 className="mt-1 font-display text-[28px] font-medium leading-tight text-ink">{place ? place.name : "Choose waters on My Coast"}</h1>
        {place ? (
          <>
            <p className="mt-1 text-[15px] text-ink-2">
              {activityLabel(audience)} · {species.label}
            </p>
            <SinceLastCheck />
            <p className="mt-4">
              <Link href={coastHref({ place: place.key, for: audience, species: species.id })} className="inline-flex min-h-[44px] items-center text-[15px] font-medium text-accent">
                Open the full briefing on My Coast →
              </Link>
            </p>
          </>
        ) : (
          <p className="mt-3">
            <Link href="/coast" className="inline-flex min-h-[44px] items-center text-[15px] font-medium text-accent">
              Go to My Coast →
            </Link>
          </p>
        )}
      </article>
    </AppShell>
  );
}
