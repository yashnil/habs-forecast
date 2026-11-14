import { OFFICIAL_STATUS } from "@/content/copy";
import s from "./sources.module.css";

/**
 * Where closures and advisories really come from (v15.1; was src/components/panels/OfficialStatusCard.tsx): Neritic's
 * list is transcribed and not checked by a person, so the page names the agencies, links their pages and gives their
 * hotlines. Words unchanged (copy.ts OFFICIAL_STATUS).
 */
export function OfficialSources() {
  return (
    <section className={s.official} data-testid="official-status" aria-labelledby="official-status-h">
      <h2 id="official-status-h">{OFFICIAL_STATUS.heading}</h2>
      <p>{OFFICIAL_STATUS.notTracked}</p>
      <p>{OFFICIAL_STATUS.instruction}</p>
      <ul className={s.officialLinks}>
        {OFFICIAL_STATUS.links.map((l) => (
          <li key={l.href}>
            <a href={l.href} target="_blank" rel="noreferrer">
              {l.label} ↗
            </a>
          </li>
        ))}
      </ul>
      <dl className={s.hotlines}>
        {OFFICIAL_STATUS.hotlines.map((h) => (
          <div key={h.tel}>
            <dt>{h.label}</dt>
            <dd>
              <a href={`tel:${h.tel}`}>{h.phone}</a>
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
