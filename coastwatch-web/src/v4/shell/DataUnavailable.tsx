import Link from "next/link";
import { Lockup } from "@/v4/brand/Brand";

/** The dataset could not be read: say so, and offer what still works (agency pages, data status). */
export function DataUnavailable({ error }: { error: string }) {
  return (
    <div className="env-instrument grid min-h-dvh place-items-center p-4" data-testid="data-unavailable">
      <div className="nx-deck w-[min(480px,100%)] p-6">
        <Lockup />
        <h1 className="mt-5 text-[22px] font-semibold">Neritic&rsquo;s data could not be loaded</h1>
        <p className="mt-2 text-[15px] text-n-ink-2">{error}. Nothing is shown in its place. Official closures and advisories are always on the agencies&rsquo; own pages.</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <a href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" className="nx-press nx-focus flex min-h-12 items-center rounded-lg border border-n-line-strong px-4 font-semibold">CDFW advisories</a>
          <a href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" className="nx-press nx-focus flex min-h-12 items-center rounded-lg border border-n-line-strong px-4 font-semibold">CDPH advisories</a>
          <Link href="/sources" className="nx-press nx-focus flex min-h-12 items-center rounded-lg border border-n-line-strong px-4 font-semibold">Data status</Link>
          <Link href="/" className="nx-press nx-focus flex min-h-12 items-center rounded-lg bg-n-ink px-4 font-semibold text-n-bg">Try again</Link>
        </div>
      </div>
    </div>
  );
}
