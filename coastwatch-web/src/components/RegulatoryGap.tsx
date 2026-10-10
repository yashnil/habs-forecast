/**
 * Preview builds only: the notice registry is a transcription that has not been reviewed by a
 * person and is known to be missing current notices. This says so on every page, names the
 * known gap, and links straight to the agencies. It never marks anything verified.
 * Sources checked 2026-10-10: CDFW health advisories page and CDPH release SN26-020.
 */
export const GAP = {
  short: "Notice list incomplete.",
  known:
    "On Oct 9, 2026 CDFW closed the recreational razor clam fishery in Del Norte County and CDPH warned against eating sport-harvested razor clams from Del Norte County. Neither is in this preview's list, which may miss other current notices.",
  links: [
    { label: "CDFW health advisories and closures", href: "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" },
    { label: "CDPH warning SN26-020", href: "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-020.aspx" },
    { label: "All CDPH shellfish advisories", href: "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" },
  ],
};

const A = ({ href, children }: { href: string; children: React.ReactNode }) => (
  <a href={href} target="_blank" rel="noreferrer" className="font-semibold underline underline-offset-2">
    {children}
  </a>
);

export function RegulatoryGapBanner() {
  return (
    <div role="alert" data-testid="regulatory-gap" className="border-b border-[#e9a8a0] bg-[#fdecea] px-4 py-2 text-[13px] leading-snug text-[#5c1a12]">
      <p className="mx-auto hidden max-w-[1400px] sm:block">
        <b className="font-semibold">⚠ {GAP.short}</b> Not an official or complete list of closures. {GAP.known}{" "}
        <span className="whitespace-nowrap">
          Check <A href={GAP.links[0].href}>CDFW</A> and <A href={GAP.links[1].href}>CDPH</A> before harvesting or eating seafood.
        </span>
      </p>
      <p className="sm:hidden">
        <b className="font-semibold">⚠ {GAP.short}</b> Missing the Oct 9 Del Norte razor clam closure and warning. Check <A href={GAP.links[0].href}>CDFW</A> and <A href={GAP.links[1].href}>CDPH</A>.
      </p>
    </div>
  );
}

export function RegulatoryGapCard() {
  return (
    <section role="alert" data-testid="regulatory-gap-card" className="space-y-1.5 rounded-lg border border-[#e9a8a0] bg-[#fdecea] p-3 text-[13px] leading-snug text-[#5c1a12]">
      <p className="font-semibold">This list is incomplete and has not been reviewed by a person.</p>
      <p>{GAP.known}</p>
      <ul className="list-disc space-y-0.5 pl-4">
        {GAP.links.map((l) => (
          <li key={l.href}>
            <A href={l.href}>{l.label}</A>
          </li>
        ))}
      </ul>
    </section>
  );
}
