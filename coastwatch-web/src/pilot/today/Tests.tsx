"use client";

import { ageWords, dayWords, type TestSiteView } from "@/pilot/facts";
import { Fold } from "@/v4/motion";
import { ToxinChart, ToxinStrip } from "./Charts";
import { Chapter, factProps } from "./ui";

/** Keeps the newest site per species for the glance strip (the tests list is one per site, newest first). */
function perSpecies(tests: TestSiteView[]) {
  const seen = new Set<string>();
  return tests.filter((t) => t.latest && !seen.has(t.site.species) && (seen.add(t.site.species), true));
}

/** Toxin tests, chapter three: the plain sentence, one dot strip against the limit; every site and sample fold away. */
export function ToxinTests({ tests, error, now, sourceUrl }: { tests: TestSiteView[]; error: boolean; now: string; sourceUrl: string | null }) {
  const caution = tests.find((t) => t.fact.detail && /still (delayed|closed)/.test(t.fact.detail))?.fact.detail?.match(/[^.]*still (?:delayed|closed)[^.]*\./)?.[0];
  return (
    <Chapter id="tests" area="tests" title="Toxin tests" testid="today-tests" say={error ? (
      <span data-testid="tests-unavailable">
        CDPH&rsquo;s domoic acid results are not available in Neritic right now. Read them on{" "}
        <a className="nx-link nx-focus" href="https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx" target="_blank" rel="noreferrer">
          CDPH&rsquo;s page
        </a>
        .
      </span>
    ) : !tests.length ? (
      <span data-testid="tests-none">CDPH has published no crab or seafood test for the waters near this port in Neritic&rsquo;s copy of its results.</span>
    ) : (
      <span data-testid="tests-lede">{tests[0].sentence}</span>
    )}>
      {tests.length > 0 && (
        <>
          <ToxinStrip sites={perSpecies(tests)} now={now} />
          {caution && <p className="nx-margin mt-2">{caution}</p>}
          <Fold testid="tests-more" title={`Every site and sample${tests.length > 1 ? ` (${tests.length} sites)` : ""}`} className="today-fold" heading="h3">
            <ul>
              {tests.map((t) => (
                <li key={t.site.id} className="border-t border-n-line py-4 first:border-t-0" data-testid={`test-${t.site.id}`} {...factProps(t.fact.id)}>
                  <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                    <h4 className="text-[16px] font-semibold text-n-ink">
                      {t.site.species} <span className="font-normal text-n-ink-2">· {t.site.bma ? `BMA ${t.site.bma}${t.site.bma_label ? ` (${t.site.bma_label})` : ""}, ` : ""}{t.site.site_label}</span>
                    </h4>
                    {t.latest && (
                      <span className={`text-[13px] ${t.stale ? "text-n-stale" : "text-n-ink-3"}`} data-testid="test-age">
                        latest set {ageWords(t.ageDays ?? 0)}
                        {t.stale ? " (old)" : ""}
                      </span>
                    )}
                  </div>
                  {t !== tests[0] && <p className="mt-1 text-[15px] leading-snug text-n-ink-2">{t.sentence}</p>}
                  {t.latest && (
                    <div className="mt-2 max-w-[520px]">
                      <ToxinChart sets={t.site.sets} action={t.action} label={`${t.site.species}, ${t.site.site_label}`} now={now} />
                      {t.latest.tissue !== "unknown" && <p className="today-src mt-1">{t.latest.tissue}</p>}
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
                </li>
              ))}
            </ul>
            <p className="today-src mt-2 flex flex-wrap gap-x-3">
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
                not detected or below the reporting limit
              </span>
            </p>
            <p className="today-src mt-1.5">
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
          </Fold>
        </>
      )}
    </Chapter>
  );
}
