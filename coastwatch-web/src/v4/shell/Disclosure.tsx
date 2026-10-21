import Link from "next/link";

/**
 * Every workspace says what Neritic is and that the notices are not verified, with the agencies one
 * tap away (#60): a research preview, not an official source; recorded test data is labelled as such.
 * v7: one wording everywhere, said once per page and quietly (neutral ink, one official tick), so the
 * page's own content leads; the full statement is the first line of the notices panel. On a phone it
 * is one short line. On a short screen (max-height 500px) it folds into the notices pill, which then
 * says "Preview" (theme.css), and the agencies are one tap away in its panel.
 */
export function RegistryDisclosure({ className = "", fixture = false, tone = "page" }: { className?: string; fixture?: boolean; tone?: "page" | "map"; compact?: boolean }) {
  // each agency link is a full 44 px target (min-w-11), at 320 px too
  const link = "nx-focus -my-3 inline-flex min-h-11 min-w-11 items-center justify-center px-1 max-[359px]:px-0.5 font-semibold text-n-ink underline decoration-n-line-strong underline-offset-[3px] hover:decoration-n-ink";
  return (
    <p className={`nx-disclosure flex flex-wrap items-center gap-x-1 text-[12.5px] leading-[18px] text-n-ink-2 max-[359px]:text-[12px] ${tone === "map" ? "nx-on-map" : ""} ${className}`} data-testid="registry-disclosure">
      {fixture && (
        <span className="mr-1 inline-flex rounded-full border border-n-official-line px-2 text-[12px] font-semibold text-n-official" role="status" data-testid="fixture-banner">
          Test data<span className="sr-only">: recorded fixture data, not a live feed</span>
        </span>
      )}
      <span className="max-md:hidden">
        <Link href="/about" className="nx-focus font-semibold text-n-ink hover:underline" data-testid="preview-disclosure">
          Research preview
        </Link>
        <span aria-hidden> · </span>
        <span aria-hidden className="mr-1 inline-block h-2 w-2 -translate-y-px rounded-full bg-n-official align-middle" />
        <span className="font-semibold text-n-ink">Notices not verified</span>, may be incomplete<span aria-hidden> ·</span>
      </span>
      {/* a phone: one short line, the words only (About is in More; a second link would crowd the agencies') */}
      <span className="md:hidden">
        <span className="font-semibold text-n-ink" data-testid="preview-disclosure-short">
          Preview
        </span>
        <span aria-hidden> · </span>
        <span aria-hidden className="mr-1 inline-block h-2 w-2 -translate-y-px rounded-full bg-n-official align-middle" />
        <span className="font-semibold text-n-ink">Notices not verified</span>
        <span aria-hidden> ·</span>
      </span>
      <span className="sr-only">Check the agencies:</span>
      <a className={link} href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noreferrer">
        CDFW
      </a>
      <span aria-hidden>·</span>
      <a className={link} href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noreferrer">
        CDPH
      </a>
    </p>
  );
}
