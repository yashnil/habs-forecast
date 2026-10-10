"use client";

import type { OfficialDataset } from "@/generated/official";
import { useOfficialDrawer } from "@/components/shell/OfficialShell";
import { Icon } from "@/components/ui/Icon";

/**
 * Preview builds only: the notice registry is a transcription that has not been reviewed by a
 * person. One strip under the masthead says so, names the newest known notices (the Oct 9, 2026
 * Del Norte razor clam closure and CDPH warning SN26-020, checked on the agency pages on
 * 2026-10-10) and whether the published list contains them yet, and links to the agencies.
 * It never marks anything verified.
 */
export const DEL_NORTE_IDS = ["cdfw-2026-razor-clam-del-norte", "cdph-2026-sn26-020-razor-clam-del-norte"] as const;

export const GAP_LINKS = [
  { label: "CDFW health advisories and closures", short: "CDFW", href: "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" },
  { label: "CDPH warning SN26-020 (Del Norte razor clams)", short: "CDPH", href: "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-020.aspx" },
  { label: "All CDPH shellfish advisories", short: "CDPH advisories", href: "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" },
];

/** True when the published registry lists both Del Norte records (still unverified). */
export function delNorteListed(official: OfficialDataset | null): boolean {
  const ids = new Set((official?.registry.records ?? []).map((r) => r.id));
  return DEL_NORTE_IDS.every((id) => ids.has(id));
}

export function gapSentence(official: OfficialDataset | null): string {
  return delNorteListed(official)
    ? "On Oct 9, 2026 CDFW closed recreational razor clam harvest in Del Norte County and CDPH warned against eating them. Both are in this list, unverified."
    : "On Oct 9, 2026 CDFW closed recreational razor clam harvest in Del Norte County and CDPH warned against eating them. Neither is in this list yet.";
}

const A = ({ href, children }: { href: string; children: React.ReactNode }) => (
  <a href={href} target="_blank" rel="noreferrer" className="font-semibold underline underline-offset-2">
    {children}
  </a>
);

export function RegulatoryGapBanner({ official }: { official: OfficialDataset | null }) {
  const { openDrawer } = useOfficialDrawer();
  const listed = delNorteListed(official);
  return (
    <div role="status" data-testid="regulatory-gap" data-del-norte={listed ? "listed" : "missing"} className="border-b border-official-line bg-official-bg px-4 py-1.5 text-[13px] leading-snug text-official-ink">
      <p className="mx-auto flex max-w-[1400px] items-baseline gap-2">
        <Icon name="shield" className="h-3.5 w-3.5 shrink-0 translate-y-[2px] text-official" />
        <span className="min-w-0">
          <b className="font-semibold">Notices here are not verified and may be incomplete.</b>{" "}
          <span className="hidden md:inline">{gapSentence(official)} </span>
          <span className="md:hidden">Del Norte razor clams closed Oct 9{listed ? "" : ", not yet listed"}. </span>
          Check <A href={GAP_LINKS[0].href}>CDFW</A> and <A href={GAP_LINKS[1].href}>CDPH</A>
          <span className="hidden md:inline"> before harvesting or eating seafood</span>.{" "}
          <button type="button" onClick={(e) => openDrawer(e.currentTarget)} className="hidden font-semibold underline underline-offset-2 md:inline">
            See the list
          </button>
        </span>
      </p>
    </div>
  );
}

export function RegulatoryGapCard({ official }: { official: OfficialDataset | null }) {
  return (
    <section role="note" data-testid="regulatory-gap-card" className="space-y-1.5 rounded-lg border border-official-line bg-official-bg p-3 text-[13px] leading-snug text-official-ink">
      <p className="font-semibold">This list is transcribed by CoastWatch, has not been checked by a person, and may miss current notices.</p>
      <p>{gapSentence(official)}</p>
      <ul className="list-disc space-y-0.5 pl-4">
        {GAP_LINKS.map((l) => (
          <li key={l.href}>
            <A href={l.href}>{l.label}</A>
          </li>
        ))}
      </ul>
    </section>
  );
}
