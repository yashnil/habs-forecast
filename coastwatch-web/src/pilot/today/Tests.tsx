"use client";

import { useState } from "react";
import { ageWords, dayWords, type TestSiteView } from "@/pilot/facts";
import { ToxinChart } from "./Charts";
import { Section } from "./ui";

const SHOWN = 2;

/** CDPH crab and seafood toxin tests: the plain sentence first, then every sample as a dot against the action level. */
export function ToxinTests({ tests, error, now, sourceUrl }: { tests: TestSiteView[]; error: boolean; now: string; sourceUrl: string | null }) {
  const [all, setAll] = useState(false);
  const list = all ? tests : tests.slice(0, SHOWN);
  return (
    <Section id="tests" area="tests" title="Crab & seafood toxin tests">
      {error ? (
        <p className="today-say" data-testid="tests-unavailable">
          CDPH&rsquo;s domoic acid results are not available in Neritic right now. Read them on{" "}
          <a className="nx-focus underline underline-offset-4" href="https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx" target="_blank" rel="noreferrer">
            CDPH&rsquo;s page
          </a>
          .
        </p>
      ) : !tests.length ? (
        <p className="today-say" data-testid="tests-none">
          CDPH has published no crab or seafood test for the waters near this port in Neritic&rsquo;s copy of its results.
        </p>
      ) : (
        <>
          <p className="today-say" data-testid="tests-lede">
            {tests[0].sentence}
          </p>
          <ul className="mt-2">
            {list.map((t) => (
              <li key={t.site.id} className="border-t border-n-line py-4 first:border-t-0" data-testid={`test-${t.site.id}`}>
                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <h3 className="text-[16px] font-semibold text-n-ink">
                    {t.site.species} <span className="font-normal text-n-ink-2">· {t.site.bma ? `BMA ${t.site.bma}, ` : ""}{t.site.site_label}</span>
                  </h3>
                  {t.latest && (
                    <span className={`text-[13px] ${t.stale ? "t-amber" : "text-n-ink-3"}`} data-testid="test-age">
                      latest set {ageWords(t.ageDays ?? 0)}
                      {t.stale ? " · old" : ""}
                    </span>
                  )}
                </div>
                {t !== tests[0] && <p className="mt-1 text-[15px] leading-snug text-n-ink-2">{t.sentence}</p>}
                {t.latest && (
                  <div className="mt-2 max-w-[460px]">
                    <ToxinChart sets={t.site.sets} action={t.action} label={`${t.site.species}, ${t.site.site_label}`} now={now} />
                    <p className="today-src mt-1 flex flex-wrap gap-x-3">
                      <span>
                        <svg width="8" height="8" aria-hidden className="mr-1 inline">
                          <circle cx="4" cy="4" r="3.2" fill="var(--nx-measured)" />
                        </svg>
                        one sample
                      </span>
                      <span>
                        <svg width="14" height="8" aria-hidden className="mr-1 inline">
                          <line x1="0" x2="14" y1="4" y2="4" stroke="var(--nx-ink)" strokeWidth="1.5" />
                        </svg>
                        set average
                      </span>
                      <span>
                        <svg width="8" height="8" aria-hidden className="mr-1 inline">
                          <circle cx="4" cy="4" r="2.6" fill="none" stroke="var(--nx-measured)" strokeWidth="1.2" />
                        </svg>
                        not detected
                      </span>
                      {t.latest.tissue !== "unknown" && <span>{t.latest.tissue}</span>}
                    </p>
                  </div>
                )}
                {t.showProgress && (
                  <div className="mt-2 flex items-start gap-3" data-testid="clean-progress">
                    <span className="mt-1 flex gap-1" aria-hidden>
                      {[0, 1].map((i) => (
                        <span key={i} className={`block h-3 w-6 rounded-[2px] border ${i < Math.min(2, t.cleanStreak) ? "border-[var(--nx-measured)] bg-[var(--nx-measured)]" : "border-n-line-strong"}`} />
                      ))}
                    </span>
                    <p className="text-[14px] leading-snug text-n-ink-2">{t.progressWords}</p>
                  </div>
                )}
                {t.fact.detail && /still (delayed|closed)/.test(t.fact.detail) && <p className="nx-margin mt-2">{t.fact.detail.match(/[^.]*still (?:delayed|closed)[^.]*\./)?.[0]}</p>}
              </li>
            ))}
          </ul>
          {tests.length > SHOWN && (
            <button type="button" onClick={() => setAll((a) => !a)} className="nx-focus nx-tint inline-flex min-h-11 items-center rounded-[3px] px-1 text-[14.5px] text-n-ink underline decoration-n-line-strong underline-offset-4" aria-expanded={all} data-testid="tests-more">
              {all ? "Show fewer sites" : `Show ${tests.length - SHOWN} more site${tests.length - SHOWN > 1 ? "s" : ""}`}
            </button>
          )}
          <p className="today-src mt-2">
            Measured by CDPH in crab and fish tissue, dated by collection. Never extrapolated. Source:{" "}
            {sourceUrl ? (
              <a href={sourceUrl} target="_blank" rel="noreferrer" className="nx-focus">
                CDPH domoic acid results
              </a>
            ) : (
              "CDPH"
            )}
            {tests[0].latest ? `, newest collection ${dayWords(tests.map((t) => t.latest?.date ?? "").sort().at(-1)!, now)}` : ""}.
          </p>
        </>
      )}
    </Section>
  );
}
