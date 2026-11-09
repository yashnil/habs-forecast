"use client";

import type { BloomView, WhaleView } from "@/pilot/facts";
import { OUTLOOK_SIGNAL_RATIO, ageWords, dayWords, factorWords, mgWords, pacificDay, rangeWords } from "@/pilot/facts";
import { BloomTimeline, OutlookScale, ZoneLadder } from "./Charts";
import { Section, SourceLink, Unverified, factProps } from "./ui";

/**
 * Bloom outlook, read top-down as two separate things that are never combined:
 * (a) NOAA's short-range model of toxin conditions in the water (C-HARM, days 0–3), with its run date;
 * (b) Neritic's 8-day chlorophyll outlook, said as higher than, about, or lower than usual for the season.
 */
export function BloomOutlook({ bloom, now, label, place }: { bloom: BloomView; now: string; label: string; place: string }) {
  const c = bloom.charm;
  const old = c?.state === "historical";
  const o = bloom.outlook;
  const read = bloom.outlookRead;
  const chart = (
    <div className="mt-3">
      <BloomTimeline days={bloom.days} today={pacificDay(now)} old={!!old} now={now} />
      <p className="today-src mt-1 flex flex-wrap gap-x-4">
        <span>
          <svg width="10" height="10" aria-hidden className="mr-1 inline">
            <rect width="10" height="10" rx="1.5" fill="var(--nx-model)" />
          </svg>
          particulate domoic acid &gt; 500 ng/L
        </span>
        <span>
          <svg width="10" height="10" aria-hidden className="mr-1 inline">
            <rect width="10" height="10" rx="1.5" fill="color-mix(in srgb, var(--nx-model) 38%, transparent)" />
          </svg>
          <i>Pseudo-nitzschia</i> &gt; 10,000 cells/L
        </span>
        <span>
          <svg width="14" height="10" aria-hidden className="mr-1 inline">
            <line x1="0" x2="14" y1="5" y2="5" stroke="var(--nx-ink-3)" strokeDasharray="2 3" />
          </svg>
          50%
        </span>
      </p>
    </div>
  );
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
  return (
    <Section id="bloom" area="bloom" title="Bloom outlook" feedback={{ factId: bloom.charmFact.id, placeId: place }}>
      <div {...factProps(bloom.charmFact.id)} data-testid="bloom-charm">
        <h3 className="today-h3 mt-4 flex flex-wrap items-baseline justify-between gap-x-3">
          <span>NOAA toxin-conditions model · 3 days</span>
          {c?.issued && <span data-testid="charm-run">run {dayWords(c.issued, now)}</span>}
        </h3>
        <p className="today-say !mt-1.5" data-testid="bloom-lede">
          {bloom.charmFact.headline}
        </p>
        {c && c.state === "stale" && c.issued && (
          <p className="mt-2 text-[14px] text-n-ink-2" data-testid="charm-stale">
            C-HARM run of {dayWords(c.issued, now)}, {ageWords(c.ageDays ?? 0)}. NOAA has not published a newer run; the days shown are that run&rsquo;s own dates.
          </p>
        )}
        {old && c?.issued ? (
          // too old to describe today's water: said once, in amber; the old run is still there for anyone who wants it
          <>
            <p className="t-amber mt-2 text-[14px]" data-testid="charm-stale">
              That run is {ageWords(c.ageDays ?? 0).replace(" ago", "")} old. Neritic shows no bloom forecast for today rather than an old one.
            </p>
            <details className="today-more mt-2" data-testid="charm-old-run">
              <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center text-[14.5px] text-n-ink underline decoration-n-line-strong underline-offset-4">Show the old run</summary>
              {chart}
            </details>
          </>
        ) : (
          chart
        )}
        <p className="mt-3 text-[14px] leading-relaxed text-n-ink-2">
          <b className="font-semibold text-n-ink">C-HARM</b> is NOAA&rsquo;s model probability of bloom and toxin <i>conditions in the water</i> for the next three days, not toxin in crab or shellfish. Its 3 km cells often stop short of harbours; near {label} it is the median of its cells within 15 km.{" "}
          <SourceLink name="NOAA C-HARM" url={c?.url ?? "https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/california/"}>
            NOAA C-HARM
          </SourceLink>
          .
        </p>
      </div>
      {bloom.outlookFact && o ? (
        <div className="mt-6 border-t border-n-line pt-1" data-testid="outlook" data-level={read?.level ?? "none"} {...factProps(bloom.outlookFact.id)}>
          <h3 className="today-h3 mt-4 flex flex-wrap items-baseline justify-between gap-x-3">
            <span>8-day chlorophyll outlook · not toxin</span>
            <span>Neritic</span>
          </h3>
          <p className="today-say !mt-1.5" data-testid="outlook-say">
            {say(bloom.outlookFact.headline, read?.phrase ?? null)}
          </p>
          {read && (
            <div className="mt-3">
              <OutlookScale read={read} errorFactor={bloom.errorFactor} signal={OUTLOOK_SIGNAL_RATIO} />
              <p className="today-src mt-1 flex flex-wrap gap-x-4">
                <span>
                  <svg width="10" height="10" aria-hidden className="mr-1 inline">
                    <circle cx="5" cy="5" r="4.5" fill="var(--nx-model)" />
                  </svg>
                  outlook, {rangeWords(o.valid_from, o.valid_to, now)}
                </span>
                {bloom.errorFactor != null && (
                  <span>
                    <svg width="14" height="10" aria-hidden className="mr-1 inline">
                      <rect width="14" height="10" rx="1.5" fill="color-mix(in srgb, var(--nx-model) 24%, transparent)" />
                    </svg>
                    typical error, a factor of {factorWords(bloom.errorFactor)} either way
                  </span>
                )}
              </p>
            </div>
          )}
          <p className="mt-3 text-[14px] leading-relaxed text-n-ink-2" data-testid="outlook-words">
            {read ? bloom.outlookHonest : "Chlorophyll is algae of every kind. One 8-day mean, not a day-by-day forecast."} {bloom.outlookUse}
            {!read && o.value != null ? " This data file carries no usual value for the season, so Neritic cannot say whether that is high or low." : ""}
            {bloom.outlookStale ? " Its window has passed." : ""}
          </p>
          <details className="today-more mt-1" data-testid="outlook-skill">
            <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center text-[14.5px] text-n-ink underline decoration-n-line-strong underline-offset-4">How good is this outlook?</summary>
            <div className="grid gap-2 pb-2 text-[14px] leading-relaxed text-n-ink-2">
              <p>
                It was tested on years it was not fitted to, against two plain guesses: &ldquo;the usual for the season&rdquo; (climatology) and &ldquo;the same as the last 8 days&rdquo; (persistence). {bloom.skillWords}
              </p>
              {read && o.value != null && o.typical != null && (
                <p data-testid="outlook-numbers">
                  Here: about {mgWords(o.value)} mg/m³ against a usual {mgWords(o.typical)} mg/m³{o.basis === "nearby" ? `, each the median of the water within 15 km of ${label}` : ""}.{o.typical_basis ? ` ${o.typical_basis}` : ""}
                </p>
              )}
              {o.caveats.length > 0 && (
                <ul className="grid list-disc gap-1 pl-5">
                  {o.caveats.slice(0, 4).map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
              )}
              <p>
                <SourceLink name="Research" url="/research">
                  How the outlook is made
                </SourceLink>
                .
              </p>
            </div>
          </details>
        </div>
      ) : (
        <p className="mt-6 border-t border-n-line pt-4 text-[14px] leading-relaxed text-n-ink-2" data-testid="outlook-missing">
          The 8-day chlorophyll outlook is not available right now.
        </p>
      )}
    </Section>
  );
}

/** Whale-entanglement rules (CDFW RAMP) for the zone the port is in. */
export function WhaleRules({ whale, now, place }: { whale: WhaleView | null; now: string; place: string }) {
  return (
    <Section id="whale" area="whale" title="Whale rules (RAMP)" feedback={whale ? { factId: whale.fact.id, placeId: place } : null}>
      {!whale ? (
        <p className="today-say" data-testid="whale-unavailable">
          Neritic has no current whale-entanglement (RAMP) record. See{" "}
          <a className="nx-focus underline underline-offset-4" href="https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries" target="_blank" rel="noreferrer">
            CDFW Whale Safe Fisheries
          </a>
          .
        </p>
      ) : (
        <div className="mt-3 flex gap-4" {...factProps(whale.fact.id)}>
          <ZoneLadder zone={whale.zone} severity={whale.fact.severity} />
          <div className="min-w-0 flex-1">
            <p className="text-[17px] leading-snug text-n-ink" data-testid="whale-lede">
              Zone {whale.zone}, {whale.label}: <b className="font-semibold">{(whale.action ?? "no action on record").replace(/\.+$/, "")}</b>.
            </p>
            <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[14.5px]">
              <dt className="text-n-ink-3">Depth limit</dt>
              <dd className="text-n-ink">{whale.depth_limit_fathoms != null ? `${whale.depth_limit_fathoms} fathoms` : "none on record"}</dd>
              <dt className="text-n-ink-3">Gear reduction</dt>
              <dd className="text-n-ink">{whale.gear_reduction_pct != null ? `${whale.gear_reduction_pct}% fewer traps` : "none on record"}</dd>
              {whale.last_assessment && (
                <>
                  <dt className="text-n-ink-3">Last assessment</dt>
                  <dd className="text-n-ink">{dayWords(whale.last_assessment, now)}</dd>
                </>
              )}
              <dt className="text-n-ink-3">Next assessment</dt>
              <dd className="text-n-ink" data-testid="whale-next">
                {whale.next_assessment ? dayWords(whale.next_assessment, now) : "not announced"}
                {whale.next_window && <span className="block text-[13px] text-n-ink-3">CDFW: {whale.next_window}</span>}
              </dd>
            </dl>
            <p className="today-src mt-2 flex flex-wrap items-center gap-2">
              <span>
                CDFW Risk Assessment Mitigation Program{whale.season ? `, ${whale.season} season` : ""} ·{" "}
                <SourceLink name="source" url={whale.source_url}>
                  source
                </SourceLink>
              </span>
              <Unverified verified={whale.verified} />
            </p>
            <p className="nx-margin mt-2">About whale entanglement in trap gear, not toxin. Neritic has no whale-sighting feed for these waters.</p>
          </div>
        </div>
      )}
    </Section>
  );
}
