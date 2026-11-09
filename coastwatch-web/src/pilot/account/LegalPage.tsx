import type { ReactNode } from "react";
import { PilotHero, PilotShell } from "@/pilot/shell/PilotShell";
import { loadTrust } from "@/lib/data";
import { LEGAL_EFFECTIVE, LEGAL_VERSION } from "./consent";

/**
 * /privacy, /terms and /sms-terms in the pilot's shell (the same top bar, dock and Almanac ground as
 * Today): a draft note (until the owner and a lawyer have read them), the effective date, the body
 * as long reading, and the attribution the CC BY licence asks for. The pilot footer links the three
 * pages to each other and back to Today and the account page.
 */
export const OWNER_CONTACT = "[OWNER: add a contact email before launch]";

export async function LegalPage({ title, testId, children }: { title: string; testId: string; children: ReactNode }) {
  return (
    <PilotShell trust={await loadTrust()}>
      <article data-testid={testId}>
        <PilotHero eyebrow="Neritic private pilot · legal" title={title}>
          <p className="mt-3 text-[14px] text-n-ink-2" data-testid="legal-effective">
            Effective {LEGAL_EFFECTIVE} · version {LEGAL_VERSION}
          </p>
          <p role="note" className="pilot-note amber mt-4" data-testid="legal-draft-banner">
            <strong className="font-semibold">Draft — pending owner/legal review.</strong> This text has not yet been reviewed by a lawyer and is not legal advice.
          </p>
        </PilotHero>
        <div className="today-wrap">
          <div className="pilot-prose nrt-legal pt-7">{children}</div>
          <p className="today-src mt-10 max-w-[70ch]" data-testid="legal-attribution">
            Structure and some wording adapted from{" "}
            <a href="https://github.com/basecamp/policies" rel="noreferrer" className="nx-focus">
              Basecamp&rsquo;s open-source policies
            </a>
            , licensed{" "}
            <a href="https://creativecommons.org/licenses/by/4.0/" rel="noreferrer" className="nx-focus">
              CC BY 4.0
            </a>
            ; changed substantially for Neritic.
          </p>
        </div>
      </article>
    </PilotShell>
  );
}
