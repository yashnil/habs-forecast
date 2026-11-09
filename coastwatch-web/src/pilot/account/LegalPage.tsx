import Link from "next/link";
import type { ReactNode } from "react";
import { LEGAL_EFFECTIVE, LEGAL_VERSION } from "./consent";

/**
 * The frame of /privacy, /terms and /sms-terms: a draft banner (until the owner and a lawyer have
 * read them), the effective date, the body, and the attribution the CC BY licence asks for.
 */
export const OWNER_CONTACT = "[OWNER: add a contact email before launch]";

export function LegalPage({ title, testId, children }: { title: string; testId: string; children: ReactNode }) {
  return (
    <article className="mx-auto w-full max-w-[760px] px-4 pb-20 pt-8 sm:px-8 sm:pt-12" data-testid={testId}>
      <p role="note" className="mb-6 rounded-xl border border-n-stale bg-n-surface px-4 py-3 text-[14.5px] font-medium leading-relaxed text-n-ink" data-testid="legal-draft-banner">
        Draft — pending owner/legal review. This text has not yet been reviewed by a lawyer and is not legal advice.
      </p>
      <header className="space-y-2">
        <h1 className="font-display text-[34px] font-medium leading-[1.1] text-n-ink sm:text-[42px]">{title}</h1>
        <p className="text-[14px] text-n-ink-2" data-testid="legal-effective">
          Effective {LEGAL_EFFECTIVE} · version {LEGAL_VERSION}
        </p>
      </header>
      <div className="nrt-legal mt-8 space-y-4 text-[16px] leading-relaxed text-n-ink-2 [&_a]:text-n-accent [&_a]:underline [&_a]:underline-offset-2 [&_h2]:pt-4 [&_h2]:font-display [&_h2]:text-[23px] [&_h2]:font-medium [&_h2]:leading-tight [&_h2]:text-n-ink [&_h3]:pt-2 [&_h3]:text-[16px] [&_h3]:font-semibold [&_h3]:text-n-ink [&_li]:mt-1 [&_strong]:text-n-ink [&_ul]:list-disc [&_ul]:pl-5">
        {children}
      </div>
      <footer className="mt-10 space-y-2 border-t border-n-line pt-6 text-[13.5px] text-n-ink-2">
        <p>
          <Link href="/privacy">Privacy Policy</Link> · <Link href="/terms">Terms of Use</Link> · <Link href="/sms-terms">SMS Terms</Link> · <Link href="/account">Your ports and alerts</Link>
        </p>
        <p data-testid="legal-attribution">
          Structure and some wording adapted from{" "}
          <a href="https://github.com/basecamp/policies" rel="noreferrer">
            Basecamp&rsquo;s open-source policies
          </a>
          , licensed{" "}
          <a href="https://creativecommons.org/licenses/by/4.0/" rel="noreferrer">
            CC BY 4.0
          </a>
          ; changed substantially for Neritic.
        </p>
      </footer>
    </article>
  );
}
