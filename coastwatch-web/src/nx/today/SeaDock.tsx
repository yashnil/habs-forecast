"use client";

import { Button } from "@/nx/ds/Button";
import { Dock } from "@/nx/ds/stage/Dock";
import { Stamp } from "@/nx/ds/Stamp";
import { Glyph, Micro, Pennant } from "@/nx/ds/synth/Grammar";
import { Skeleton } from "@/nx/ds/DataState";
import type { GateFact } from "./gates";
import { SeaBars } from "@/nx/ds/synth/charts";
import type { SeaBarsModel } from "./ticket";
import { pennantsFor } from "./ticket";
import s from "./today.module.css";

export type SeaDockProps = {
  state: "loading" | "error" | "ok";
  zone: string | null;
  zoneName: string | null;
  advisories: GateFact[];
  bars: SeaBarsModel | null;
  tide: string | null;
  now: string;
  /** a phone shows the advisories in the ticket's strip; the dock does not repeat them */
  phone: boolean;
  /** the trip from this port, when the ticket's primary is not already it */
  tripHref?: string | null;
};

/** An advisory in NWS's words, split into its event ("Small Craft Advisory") and its time words ("in effect from …"). */
export function advisoryParts(a: GateFact): { event: string; when: string } {
  const event = a.short ?? a.text.replace(/^NWS\s+/, "").split(/\s+(in effect|until|through|from)\b/)[0];
  const rest = a.text.replace(/^NWS\s+/, "");
  const when = (rest.toLowerCase().startsWith(event.toLowerCase()) ? rest.slice(event.length) : rest).replace(/^[\s:]+/, "");
  return { event, when };
}

/**
 * The sea dock (v14 synthesis, Today): along the chart's foot, where time lives. On the left the NWS zone, the
 * advisories in force with their pennants and the next tide; on the right the NWS seas per period as dashed bars
 * (predicted), the buoy's one measured point and the advisories as coral spans. On a phone it is a block in the
 * sheet after the ticket.
 */
export function SeaDock({ state, zone, zoneName, advisories, bars, tide, now, phone, tripHref }: SeaDockProps) {
  return (
    <Dock label="Sea, next four days" width={840} className={s.seaDock} data-testid="nx-sea-dock">
      <div className={s.seaLeft}>
        <Micro as="h2">Sea{zone ? ` · NWS ${zone}` : ""}</Micro>
        {zoneName ? <p className={s.seaZone}>{zoneName}</p> : null}
        {state === "loading" ? <Skeleton label="Reading NWS, the buoy and the tide" lines={2} /> : null}
        {state === "error" ? <p className={s.seaZone}>NWS, the buoy and the tide could not be read just now. Check weather.gov.</p> : null}
        {!phone && advisories.length ? (
          <div className={s.advList} data-testid="advisories">
            {advisories.map((a) => {
              const p = advisoryParts(a);
              return (
                <div key={a.id} className={s.adv}>
                  <Pennant n={pennantsFor(p.event)} />
                  <div>
                    <b>{p.event}</b>
                    <span>{p.when}</span>
                  </div>
                </div>
              );
            })}
            {advisories[0].provenance ? <Stamp provenance={advisories[0].provenance} now={now} className={s.advStamp} /> : null}
          </div>
        ) : null}
        {tide ? (
          <p className={s.tide}>
            <Glyph kind="predicted" />
            Next tide {tide}
          </p>
        ) : null}
        {tripHref ? (
          <p className={s.seaMore}>
            <Button variant="quiet" href={tripHref}>
              Plan a trip
            </Button>
          </p>
        ) : null}
      </div>
      <div className={s.seaChart}>{bars ? <SeaBars {...bars} height={phone ? 132 : 128} /> : state === "ok" ? <p className={s.seaZone}>NWS gives no seas for this zone&apos;s next periods.</p> : null}</div>
    </Dock>
  );
}
