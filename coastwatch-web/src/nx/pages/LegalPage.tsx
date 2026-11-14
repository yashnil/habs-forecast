import type { ReactNode } from "react";
import { LEGAL_FRAME } from "@/content/pages";
import { LEGAL_EFFECTIVE, LEGAL_VERSION } from "@/pilot/account/consent";
import { InfoPage, infoStyles as s, type ContentsItem, type RegistryTrust } from "./InfoPage";

/**
 * /privacy, /terms and /sms-terms (v15.1: in the shell, on the design system; was src/pilot/account/LegalPage.tsx in
 * the pilot's v4 shell). The head says the effective date and that the text is a draft until the owner and a lawyer
 * have read it; the body is long reading with its contents beside it; the attribution the CC BY licence asks for
 * closes it. The bodies are word for word in src/content/legal.
 */
export function LegalPage({ id, title, testId, trust, contents, children }: { id: string; title: string; testId: string; trust: RegistryTrust; contents: ContentsItem[]; children: ReactNode }) {
  return (
    <InfoPage
      id={id}
      testId={testId}
      kicker={LEGAL_FRAME.kicker}
      title={title}
      trust={trust}
      contents={contents}
      footer="legal"
      prose
      head={
        <>
          <p className={s.small} data-testid="legal-effective">
            Effective {LEGAL_EFFECTIVE} · version {LEGAL_VERSION}
          </p>
          <p role="note" className={`${s.note} ${s.noteReg}`} data-testid="legal-draft-banner">
            <strong>{LEGAL_FRAME.draftLead}</strong> {LEGAL_FRAME.draftRest}
          </p>
        </>
      }
    >
      {children}
      <p className={`${s.small} ${s.attr}`} data-testid="legal-attribution">
        {LEGAL_FRAME.attributionLead}{" "}
        <a href={LEGAL_FRAME.attributionPoliciesUrl} rel="noreferrer">
          {LEGAL_FRAME.attributionPolicies}
        </a>
        {LEGAL_FRAME.attributionLicensed}{" "}
        <a href={LEGAL_FRAME.attributionLicenceUrl} rel="noreferrer">
          {LEGAL_FRAME.attributionLicence}
        </a>
        {LEGAL_FRAME.attributionRest}
      </p>
    </InfoPage>
  );
}
