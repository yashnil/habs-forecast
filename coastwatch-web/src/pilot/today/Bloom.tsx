"use client";

import type { BloomView, WhaleView } from "@/pilot/facts";
import { ageWords, dayWords, pacificDay } from "@/pilot/facts";
import { BloomTimeline, ZoneLadder } from "./Charts";
import { Section, SourceLink, Unverified } from "./ui";

/** Bloom outlook: C-HARM days 0–3 (probability of conditions in the water) then the 8-day chlorophyll outlook, on one timeline. */
export function BloomOutlook({ bloom, now, label }: { bloom: BloomView; now: string; label: string }) {
  const c = bloom.charm;
  const old = c?.state === "historical";
  return (
    <Section id="bloom" area="bloom" title="Bloom outlook">
      <p className="today-say" data-testid="bloom-lede">
        {bloom.charmFact.headline}
      </p>
      {c && c.state !== "current" && c.issued && (
        <p className={`mt-2 text-[14px] ${old ? "t-amber" : "text-n-ink-2"}`} data-testid="charm-stale">
          {old ? "Out of date: " : ""}C-HARM run of {dayWords(c.issued, now)}, {ageWords(c.ageDays ?? 0)}. {old ? "It does not describe today's water." : "NOAA has not published a newer run; the days shown are that run's own dates."}
        </p>
      )}
      <div className="mt-3">
        <BloomTimeline days={bloom.days} outlook={bloom.outlook} today={pacificDay(now)} old={!!old} now={now} />
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
        </p>
      </div>
      <div className="mt-4 grid gap-3 text-[14px] leading-relaxed text-n-ink-2">
        <p>
          <b className="font-semibold text-n-ink">C-HARM</b> is NOAA&rsquo;s model probability of bloom and toxin <i>conditions in the water</i> for the next three days, not toxin in crab or shellfish. Its 3 km cells often stop short of harbours; near {label} it is the median of its cells within 15 km.{" "}
          <SourceLink name="NOAA C-HARM" url={c?.url ?? "https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/california/"}>
            NOAA C-HARM
          </SourceLink>
          .
        </p>
        {bloom.outlookFact ? (
          <p data-testid="outlook-words">
            <b className="font-semibold text-n-ink">Chlorophyll outlook</b>: a bloom indicator, not a toxin forecast. {bloom.outlookFact.value ? `About ${bloom.outlookFact.value} as one mean over ${dayWords(bloom.outlook!.valid_from, now)}–${dayWords(bloom.outlook!.valid_to, now)}` : "No value near this port"}
            {bloom.outlook?.basis === "nearby" ? " (median of water within 15 km)" : ""}. Chlorophyll is algae of every kind. {bloom.skillWords}
            {bloom.outlookStale ? " Its window has passed." : ""}
          </p>
        ) : (
          <p data-testid="outlook-missing">The 8-day chlorophyll outlook is not available right now.</p>
        )}
      </div>
    </Section>
  );
}

/** Whale-entanglement rules (CDFW RAMP) for the zone the port is in. */
export function WhaleRules({ whale, now }: { whale: WhaleView | null; now: string }) {
  return (
    <Section id="whale" area="whale" title="Whale rules (RAMP)">
      {!whale ? (
        <p className="today-say" data-testid="whale-unavailable">
          Neritic has no current whale-entanglement (RAMP) record. See{" "}
          <a className="nx-focus underline underline-offset-4" href="https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries" target="_blank" rel="noreferrer">
            CDFW Whale Safe Fisheries
          </a>
          .
        </p>
      ) : (
        <div className="mt-3 flex gap-4">
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
