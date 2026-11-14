import { LIVE_SOURCES_COPY as C } from "@/content/pages";
import { DINEOF } from "@/lib/ocean/chlorophyll";
import { SOURCE as WCOFS } from "@/lib/ocean/wcofs";
import s from "./sources.module.css";

type Row = { id: string; name: string; what: string; links: Array<[string, string]>; licence: string; flag?: string };

/** The live sources (v15.1): read on request by the server, each with its attribution, licence and links. */
export function LiveSources() {
  const rows: Row[] = [
    { id: "wcofs", name: WCOFS.name, what: C.wcofsWhat, links: [[C.wcofsVia, WCOFS.url], ["About WCOFS", WCOFS.info_url]], licence: WCOFS.licence, flag: WCOFS.developmental ? "Developmental" : undefined },
    { id: "dineof", name: DINEOF.source.name, what: `${C.dineofWhat} ${C.dineofCite}.`, links: [["NOAA CoastWatch ERDDAP", DINEOF.source.url], ["Method (DOI)", DINEOF.source.info_url]], licence: DINEOF.source.licence, flag: "Estimate" },
    ...C.rows.map((r) => ({ id: r.id, name: r.name, what: r.what, links: [["Source", r.url]] as Array<[string, string]>, licence: r.licence })),
  ];
  return (
    <section aria-labelledby="live-h">
      <div className={s.tableHead}>
        <h2 id="live-h">{C.heading}</h2>
      </div>
      <p className={s.note}>{C.intro}</p>
      <ul className={s.rows} data-testid="live-sources">
        {rows.map((r) => (
          <li key={r.id} className={s.row} data-testid={`live-source-${r.id}`}>
            <div className={s.rowHead}>
              <h3>{r.name}</h3>
              {r.flag ? <span className={s.verify}>{r.flag}</span> : null}
            </div>
            <p className={s.note}>{r.what}</p>
            <p className={s.note}>
              {r.links.map(([label, href], i) => (
                <span key={href}>
                  {i ? " · " : ""}
                  <a href={href} target="_blank" rel="noreferrer" className={s.liveLink}>
                    {label} ↗
                  </a>
                </span>
              ))}
              {" · "}
              {r.licence}
            </p>
          </li>
        ))}
      </ul>
    </section>
  );
}
