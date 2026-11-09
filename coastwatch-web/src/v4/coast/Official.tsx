"use client";

import type { ReactNode } from "react";
import type { CoastPlace } from "@/lib/coast";
import { MARK_SHORT, monthDay, regionFor, regionSeasons } from "@/lib/speciesSeasons";
import { CDFW_COMMERCIAL, CDFW_INSEASON } from "@/content/seasons";
import { AGENCY_LINKS } from "@/content/coast";
import type { Species } from "@/content/species";
import { dayWords } from "@/pilot/facts";
import { Icon } from "@/v4/ui/Icon";
import { Tag } from "@/v4/ui/Tag";
import { Unverified } from "@/v4/ui/Unverified";
import type { SeasonStatus } from "./season";
import type { TripStatusLine } from "./tripStatus";

const ext = { target: "_blank", rel: "noreferrer" } as const;
const link = "nx-link nx-focus inline-flex min-h-11 items-center gap-1 sm:min-h-8";

/**
 * The trip's status, first and short (v11): Today's status word for the species the trip names, with its
 * not-verified marker, then the sport-list reading labelled as the sport list, and at every width, "Nothing
 * here says a trip is allowed". Never a statement that a trip is allowed: the sport list's reopening date is
 * never shown to a commercial reader, and the closed mark is never hidden from one.
 */
export function TripStatus({ line, season, species, place, now, details }: { line: TripStatusLine; season: SeasonStatus; species: Species; place: CoastPlace; now: string; details?: ReactNode }) {
  const region = regionFor(place.lat);
  const rs = region ? regionSeasons(region, species.id) : null;
  const { item, commercial, commercialOnly, reopen, groundfish: gf } = season;
  const unverifiedCount = line.kind === "tally" ? (() => { const on = line.rows.filter((r) => r.verified !== null); return on.length ? `${on.filter((r) => r.verified === false).length} of ${on.length}` : undefined; })() : undefined;
  // the sport list: a recreational reader's season, a commercial reader's context (its reopen date is never said to them)
  const sport = item ? (
    <>
      <b className="font-semibold">{item.name}</b>
      <span data-testid="glance-season-list">
        {" "}
        is on CDFW&rsquo;s {MARK_SHORT[item.mark].toLowerCase()}
        {commercial ? " for sport fishing" : ""}.
      </span>
    </>
  ) : rs ? (
    commercialOnly ? (
      <>
        <b className="font-semibold">{species.label}</b>: commercial only, not on CDFW&rsquo;s sport list.
      </>
    ) : species.id === "any" ? (
      `CDFW ${region!.label} sport list: ${rs.counts.open + rs.counts.alert} open, ${rs.counts.closed} closed.`
    ) : (
      <>
        <b className="font-semibold">{species.label}</b>: no entry of its own on CDFW&rsquo;s {region!.label} sport list.
      </>
    )
  ) : (
    "Outside CDFW's five sport-fishing regions."
  );
  return (
    <section aria-labelledby="official-context-h" className="coast-status" data-testid="glance-official" id="status">
      <h2 id="official-context-h" className="sr-only">
        Status for this trip, from the port's status record
      </h2>
      {line.kind === "row" ? (
        <p className="coast-status-lead" data-testid="glance-status" data-level={line.level}>
          <Tag level={line.level} testId="glance-status-word">
            {line.word}
          </Tag>
          <span className="coast-status-sentence" data-testid="glance-status-sentence">
            {line.unclear ? <b className="font-semibold">Status unclear, awaiting a person&rsquo;s review.</b> : line.sentence}
          </span>
        </p>
      ) : line.kind === "tally" ? (
        <p className="coast-status-lead" data-testid="glance-status">
          <span className="coast-status-sentence" data-testid="glance-status-sentence">
            {line.tally}. <span className="text-n-ink-2">None on record is not a promise it is open.</span>
          </span>
        </p>
      ) : (
        <p className="coast-status-lead" data-testid="glance-status">
          <Tag level="none" testId="glance-status-word" />
          <span className="coast-status-sentence" data-testid="glance-status-sentence">
            Not a promise it is open. Neritic keeps no status record for this trip here.
          </span>
        </p>
      )}
      <div className="coast-status-meta">
        {line.kind === "row" && line.row.verified === true ? <span className="text-n-ink-2">Checked by a person</span> : <Unverified count={unverifiedCount} testId="glance-not-verified" />}
        {line.kind === "row" && line.row.checked_at ? (
          <span className="text-n-ink-2" data-testid="glance-status-source">
            {line.row.source?.name ?? "Agency"}, checked {dayWords(line.row.checked_at, now)}
          </span>
        ) : null}
        <span className="text-n-ink" data-testid="glance-no-permission">
          Nothing here says a trip is allowed.
        </span>
      </div>
      <p className="coast-status-sport">
        <a href="#seasons-h" className="nx-focus hover:underline hover:decoration-n-line-strong hover:underline-offset-4" data-testid="glance-season">
          {sport}
        </a>
        {!commercial && item && reopen && <span> CDFW: &ldquo;scheduled to reopen on {reopen}&rdquo;.</span>}
        {!commercial && item && !reopen && gf && (
          <span>
            {" "}
            Groundfish {monthDay(gf.from)}–{monthDay(gf.to)}: {gf.nearshore}.
          </span>
        )}{" "}
      </p>
      {details}
    </section>
  );
}

/** What "Season details" holds above CDFW's list: the agency pages (a phone's only place for them). */
export function SeasonLinks({ place, season }: { place: CoastPlace; season: SeasonStatus }) {
  const region = regionFor(place.lat);
  const showCommercial = season.commercial || season.commercialOnly;
  return (
    <p className="flex flex-wrap items-center gap-x-4 text-[14px]">
      {showCommercial ? (
        <>
          <a href={CDFW_COMMERCIAL.url} {...ext} className={link}>
            CDFW commercial <Icon name="external" className="h-3.5 w-3.5" />
          </a>
          <a href={CDFW_INSEASON.url} {...ext} className={link}>
            In-season page <Icon name="external" className="h-3.5 w-3.5" />
          </a>
        </>
      ) : (
        <>
          {region && (
            <a href={region.url} {...ext} className={link}>
              CDFW {region.label} region <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          )}
          <a href={CDFW_INSEASON.url} {...ext} className={link}>
            In-season page <Icon name="external" className="h-3.5 w-3.5" />
          </a>
        </>
      )}
      <a href={AGENCY_LINKS.cdphAdvisories.url} {...ext} className={link}>
        CDPH advisories <Icon name="external" className="h-3.5 w-3.5" />
      </a>
    </p>
  );
}
