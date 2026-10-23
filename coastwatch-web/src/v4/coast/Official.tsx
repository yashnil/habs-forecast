"use client";

import type { ReactNode } from "react";
import type { Audience, CoastPlace } from "@/lib/coast";
import { MARK_SHORT, monthDay, regionFor, regionSeasons } from "@/lib/speciesSeasons";
import { CDFW_COMMERCIAL, CDFW_INSEASON } from "@/content/seasons";
import { AGENCY_LINKS } from "@/content/coast";
import { speciesPhrase, type Species } from "@/content/species";
import { Icon } from "@/v4/ui/Icon";
import { Caps } from "./ui/Instrument";
import type { SeasonStatus } from "./season";

const ext = { target: "_blank", rel: "noreferrer" } as const;
const link = "nx-focus inline-flex min-h-11 items-center gap-1 text-n-ink underline decoration-n-official-line underline-offset-4 hover:decoration-n-official sm:min-h-8";

/**
 * The season, first and short (v10 pass 2): two lines, the list Neritic quotes and what it says for the
 * trip's species (with the plain fact that Neritic holds no commercial season), then the agency pages and
 * "Season details", which holds CDFW's full list. The notices tied to these waters are counted once, in
 * the decision row. Never a statement that a trip is allowed: the sport list's reopening date is never
 * shown to a commercial reader (v6 P0 2), and the closed mark is never hidden from one (D3).
 */
export function OfficialContext({ place, species, season, details }: { place: CoastPlace; species: Species; audience: Audience; season: SeasonStatus; details?: ReactNode }) {
  const region = regionFor(place.lat);
  const rs = region ? regionSeasons(region, species.id) : null;
  const { item, commercial, commercialOnly, reopen, groundfish: gf } = season;
  const showCommercial = commercial || commercialOnly;
  return (
    <section aria-labelledby="official-context-h" className="flex flex-col gap-1 border-l-2 border-n-official py-0.5 pl-4" data-testid="glance-official">
      <h2 id="official-context-h">
        <Caps family="official">
          Season · CDFW sport list<span className="max-sm:hidden">, Neritic&rsquo;s copy</span> · not verified
        </Caps>
      </h2>
      <p className="max-w-[72ch] text-[15px] leading-[1.5] text-n-ink">
        <a href="#seasons-h" className="nx-focus rounded-[2px] hover:underline hover:decoration-n-line-strong hover:underline-offset-4" data-testid="glance-season">
          {item ? (
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
              `CDFW ${region!.label}: ${rs.counts.open + rs.counts.alert} on the open list, ${rs.counts.closed} closed.`
            ) : (
              <>
                <b className="font-semibold">{species.label}</b>: no entry of its own on CDFW&rsquo;s {region!.label} list.
              </>
            )
          ) : (
            "Outside CDFW's five sport-fishing regions."
          )}
        </a>
        {!commercial && item && reopen && <span className="text-n-ink-2"> CDFW: &ldquo;scheduled to reopen on {reopen}&rdquo;.</span>}
        {!commercial && item && !reopen && gf && (
          <span className="text-n-ink-2">
            {" "}
            Groundfish {monthDay(gf.from)}–{monthDay(gf.to)}: {gf.nearshore}.
          </span>
        )}
        {showCommercial && rs && (
          <span data-testid="glance-commercial">
            {" "}
            <b className="font-semibold">Neritic holds no commercial season{species.id !== "any" ? ` for ${speciesPhrase(species)}` : ""}</b>
            <span className="max-sm:hidden">: it is on CDFW&rsquo;s commercial pages</span>.
          </span>
        )}{" "}
        <span className="text-n-ink-2 max-sm:hidden">Nothing here says a trip is allowed.</span>
      </p>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-0 text-[13.5px]">
        <span className="contents max-sm:hidden">
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
        </span>
        {details}
      </div>
    </section>
  );
}

/** What "Season details" holds above CDFW's list: the agency pages (a phone's only place for them) and the plain caveat. */
export function SeasonLinks({ place, season }: { place: CoastPlace; season: SeasonStatus }) {
  const region = regionFor(place.lat);
  const showCommercial = season.commercial || season.commercialOnly;
  return (
    <p className="flex flex-wrap items-center gap-x-4 text-[13.5px] sm:hidden">
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
      <span className="basis-full text-n-ink-2">Nothing here says a trip is allowed.</span>
    </p>
  );
}
