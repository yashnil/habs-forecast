import Link from "next/link";

/**
 * Every workspace says what Neritic is and that the notices are not verified, with the agencies one
 * tap away (#60): a research preview, not an official source; recorded test data is labelled as such.
 * Three wordings: full (reading pages), `compact` (Explore, floating over the map between its panels)
 * and, on a phone, one short line. On a short screen (max-height 500px) it folds into the notices
 * pill, which then says "Preview" (theme.css), and the agencies are one tap away in its panel.
 */
export function RegistryDisclosure({ className = "", fixture = false, compact = false }: { className?: string; fixture?: boolean; compact?: boolean }) {
  // each agency link is a full 44 px target (min-w-11), at 320 px too
  const link = "nx-focus -my-2.5 inline-flex min-h-11 min-w-11 items-center justify-center px-1 max-[359px]:px-0.5 font-semibold text-n-accent underline underline-offset-2";
  return (
    <p className={`nx-disclosure text-[13.5px] leading-snug text-n-ink-2 max-[359px]:text-[12px] ${className}`} data-testid="registry-disclosure">
      {fixture && (
        <span className="mr-1.5 inline-flex rounded-full border border-n-official-line px-2 text-[12.5px] font-semibold text-n-official" role="status" data-testid="fixture-banner">
          Test data<span className="sr-only">: recorded fixture data, not a live feed</span>
        </span>
      )}
      {compact ? (
        <span className="max-md:hidden">
          <Link href="/about" className="nx-focus font-semibold text-n-ink" data-testid="preview-disclosure">
            Research preview
          </Link>{" "}
          · <span className="font-semibold text-n-official">Notices not verified</span>, may be incomplete ·
        </span>
      ) : (
        <span className="max-md:hidden">
          <Link href="/about" className="nx-focus font-semibold text-n-ink" data-testid="preview-disclosure">
            Research preview, not an official source.
          </Link>{" "}
          <span className="font-semibold text-n-official">Official notices are not verified.</span> The list may be incomplete; check
        </span>
      )}
      {/* a phone: one short line, the words only (About is in More; a second link would crowd the agencies') */}
      <span className="md:hidden">
        <span className="font-semibold text-n-ink" data-testid="preview-disclosure-short">
          Preview
        </span>{" "}
        · <span className="font-semibold text-n-official">Notices not verified</span> ·
      </span>{" "}
      <a className={link} href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noreferrer">
        CDFW
      </a>
      <span aria-hidden> · </span>
      <a className={link} href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noreferrer">
        CDPH
      </a>
    </p>
  );
}
