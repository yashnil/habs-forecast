"use client";

import type { ReactNode } from "react";
import { OFFICIAL_STATUS } from "@/content/copy";
import { Icon } from "./Icon";
import { Pop } from "./Pop";

/**
 * The not-verified marker (design.md §3, §4.3): a shield with a slash and the words, ink-2, a dotted
 * underline; never colour only. It opens the explanation. Use it wherever an unreviewed official record
 * is quoted.
 *
 * <Unverified />                      "Not verified"
 * <Unverified count="7 of 7" />       "7 of 7 not verified" (a group's computed count)
 * explanation: what is unconfirmed here; defaults to the registry's wording. CDFW and CDPH always follow it.
 */
export function Unverified({ count, explanation, className = "", testId = "not-verified" }: { count?: string; explanation?: ReactNode; className?: string; testId?: string }) {
  return (
    <Pop
      label="Not verified: what this means"
      testId={testId}
      triggerClass={`nx-unverified nx-focus ${className}`}
      trigger={
        <>
          <Icon name="shieldOff" className="h-3.5 w-3.5" />
          <span>{count ? `${count} not verified` : "Not verified"}</span>
        </>
      }
    >
      <p className="text-[15px] font-semibold text-n-ink">Not verified</p>
      <p className="mt-1 text-[14px] leading-[20px] text-n-ink-2">
        {explanation ?? <>Neritic transcribed this from the agency&rsquo;s page and a person has not yet checked it against the source. {OFFICIAL_STATUS.notVerifiedLead}</>}
      </p>
      <p className="mt-2 flex gap-4 text-[14px]">
        <a className="nx-link nx-focus inline-flex min-h-11 items-center gap-1" href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noreferrer">
          CDFW <Icon name="external" className="h-3.5 w-3.5" />
        </a>
        <a className="nx-link nx-focus inline-flex min-h-11 items-center gap-1" href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noreferrer">
          CDPH <Icon name="external" className="h-3.5 w-3.5" />
        </a>
      </p>
    </Pop>
  );
}
