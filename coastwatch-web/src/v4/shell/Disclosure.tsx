/** Every workspace says the notices are not verified, with the agencies one tap away (#60). */
export function RegistryDisclosure({ className = "" }: { className?: string }) {
  const link = "nx-focus -my-2.5 inline-flex min-h-11 items-center px-1 font-semibold text-n-accent underline underline-offset-2";
  return (
    <p className={`text-[13.5px] leading-snug text-n-ink-2 ${className}`} data-testid="registry-disclosure">
      <span className="font-semibold text-n-official">
        <span className="max-md:hidden">Official notices are not verified.</span>
        <span className="md:hidden">Notices not verified</span>
      </span>
      <span className="max-md:hidden"> The list may be incomplete; check</span>
      <span className="md:hidden"> ·</span>{" "}
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
