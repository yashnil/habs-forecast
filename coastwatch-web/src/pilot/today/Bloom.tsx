"use client";

import Link from "next/link";
import type { BloomView } from "@/pilot/facts";
import { OUTLOOK_SIGNAL_RATIO, ageWords, dayWords, factorWords, mgWords, pacificDay, rangeWords } from "@/pilot/facts";
import { Fold } from "@/v4/motion";
import { BloomAxis, BloomTimeline, OutlookScale } from "./Charts";
import { Chapter, SourceLink, factProps } from "./ui";

/**
 * Bloom, chapter four: two labelled steps on one time axis, never combined. NOAA's C-HARM (a model of
 * conditions in the water) and Neritic's 8-day chlorophyll outlook against usual. A C-HARM run whose days
 * have all passed is a past run: it steps back behind the outlook and is said as one.
 */
export function BloomOutlook({ bloom, now, label, exploreHref }: { bloom: BloomView; now: string; label: string; exploreHref: string }) {
  const c = bloom.charm;
  const old = c?.state === "historical";
  const o = bloom.outlook;
  const read = bloom.outlookRead;
  const today = pacificDay(now);
  // the plain sentence, with its three words in bold
  const say = (text: string, phrase: string | null) => {
    const i = phrase ? text.indexOf(phrase) : -1;
    return i < 0 ? (
      text
    ) : (
      <>
        {text.slice(0, i)}
        <b className="font-semibold">{phrase}</b>
        {text.slice(i + phrase!.length)}
      </>
    );
  };
  const charm = (
    <div {...factProps(bloom.charmFact.id)} data-testid="bloom-charm" data-past={bloom.charmPast ? "" : undefined} className="today-step">
      <h3 className="nx-caps flex flex-wrap items-baseline justify-between gap-x-3">
        <span>NOAA C-HARM{bloom.charmPast ? " · past run" : " · next 3 days"} · conditions in the water, not toxin in seafood</span>
        {c?.issued && <span data-testid="charm-run">run {dayWords(c.issued, now)}</span>}
      </h3>
      <p className="today-step-say" data-testid="bloom-lede">
        {bloom.charmFact.headline}
      </p>
      {old && c?.issued ? (
        <p className="today-note mt-1 text-n-stale" data-testid="charm-stale">
          That run is {ageWords(c.ageDays ?? 0).replace(" ago", "")} old. Neritic shows no bloom forecast for today rather than an old one.
        </p>
      ) : null}
    </div>
  );
  const outlook =
    bloom.outlookFact && o ? (
      <div className="today-step" data-testid="outlook" data-level={read?.level ?? "none"} {...factProps(bloom.outlookFact.id)}>
        <h3 className="nx-caps flex flex-wrap items-baseline justify-between gap-x-3">
          <span>8-day chlorophyll outlook · not toxin</span>
          <span>Neritic</span>
        </h3>
        <p className="today-step-say" data-testid="outlook-say">
          {say(bloom.outlookFact.headline, read?.phrase ?? null)}
        </p>
      </div>
    ) : (
      <p className="today-note" data-testid="outlook-missing">
        The 8-day chlorophyll outlook is not available right now.
      </p>
    );
  const pastRun = bloom.charmPast || old;
  return (
    <Chapter
      id="bloom"
      area="bloom"
      title="Bloom"
      testid="today-bloom"
      say={
        read && !bloom.outlookStale ? `Chlorophyll is expected to run ${read.phrase} for this time of year. That says nothing about toxin.${pastRun ? " NOAA's newest C-HARM run has already passed." : ""}` : pastRun ? "NOAA's newest C-HARM run has already passed." : undefined
      }
    >
      <BloomAxis days={old ? [] : bloom.days} today={today} past={!!bloom.charmPast} outlook={o ? { from: o.valid_from, to: o.valid_to, phrase: read?.phrase ?? "outlook", stale: bloom.outlookStale } : null} />
      <div className="mt-4 grid gap-5">
        {/* a past run steps back behind the outlook; a current run comes first, in time order */}
        {pastRun ? (
          <>
            {outlook}
            {charm}
          </>
        ) : (
          <>
            {charm}
            {outlook}
          </>
        )}
      </div>
      <Link href={exploreHref} className="nx-link nx-focus today-go" data-testid="link-explore">
        See it on the map
      </Link>
      <Fold testid="bloom-more" title="What these are, how sure, and their caveats" className="today-fold" heading="h3">
        <div className="grid gap-3 pb-1 text-[14px] leading-relaxed text-n-ink-2">
          <p>
            <b className="font-semibold text-n-ink">C-HARM</b> is NOAA&rsquo;s model probability of bloom and toxin <i>conditions in the water</i> for the next three days, not toxin in crab or shellfish. Bars are the probability of particulate domoic acid above 500 ng/L; its 3 km cells often stop short of harbours, so near {label} it is the median of its cells within 15 km.{" "}
            <SourceLink name="NOAA C-HARM" url={c?.url ?? "https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/california/"}>
              NOAA C-HARM
            </SourceLink>
            .
          </p>
          {c && c.state === "stale" && c.issued && (
            <p data-testid="charm-stale">
              C-HARM run of {dayWords(c.issued, now)}, {ageWords(c.ageDays ?? 0)}. NOAA has not published a newer run; the days shown are that run&rsquo;s own dates.
            </p>
          )}
          {old && c?.issued && (
            <div data-testid="charm-old-run">
              <BloomTimeline days={bloom.days} today={today} old now={now} />
            </div>
          )}
          {o && read && (
            <div>
              <OutlookScale read={read} errorFactor={bloom.errorFactor} signal={OUTLOOK_SIGNAL_RATIO} />
              <p className="today-src mt-1">
                Outlook, {rangeWords(o.valid_from, o.valid_to, now)}
                {bloom.errorFactor != null ? ` · typical error, a factor of ${factorWords(bloom.errorFactor)} either way` : ""}
              </p>
            </div>
          )}
          {bloom.outlookFact && o && (
            <p data-testid="outlook-words">
              {read ? bloom.outlookHonest : "Chlorophyll is algae of every kind. One 8-day mean, not a day-by-day forecast."} {bloom.outlookUse}
              {!read && o.value != null ? " This data file carries no usual value for the season, so Neritic cannot say whether that is high or low." : ""}
              {bloom.outlookStale ? " Its window has passed." : ""}
            </p>
          )}
          {bloom.outlookFact && o && (
            <div data-testid="outlook-skill">
              <p>
                It was tested on years it was not fitted to, against two plain guesses: &ldquo;the usual for the season&rdquo; (climatology) and &ldquo;the same as the last 8 days&rdquo; (persistence). {bloom.skillWords}
              </p>
              {read && o.value != null && o.typical != null && (
                <p className="mt-2" data-testid="outlook-numbers">
                  Here: about {mgWords(o.value)} mg/m³ against a usual {mgWords(o.typical)} mg/m³{o.basis === "nearby" ? `, each the median of the water within 15 km of ${label}` : ""}.{o.typical_basis ? ` ${o.typical_basis}` : ""}
                </p>
              )}
              {o.caveats.length > 0 && (
                <ul className="mt-2 grid list-disc gap-1 pl-5">
                  {o.caveats.slice(0, 4).map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
              )}
              <p className="mt-2">
                <SourceLink name="Research" url="/research">
                  How the outlook is made
                </SourceLink>
                .
              </p>
            </div>
          )}
        </div>
      </Fold>
    </Chapter>
  );
}
