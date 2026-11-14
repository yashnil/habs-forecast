"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { KEPT_COPY } from "@/content/pages";
import { AGENCY_LINKS } from "@/content/coast";
import { coastHref, parseAudience } from "@/lib/coast";
import { KEPT_WARNING, deleteKept, keptFilename, keptHtml, keptSections, keptTitle, readKept, stamp, type Kept } from "@/lib/keptBriefing";
import { Button } from "@/nx/ds/Button";
import { InfoPage, infoStyles as i } from "./InfoPage";
import s from "./kept.module.css";

/**
 * Briefings kept on this device (v15.1; was src/components/coast/KeptView.tsx): dated copies to read, print or save
 * as a file. Read from the device only, and marked as a kept copy so it is never mistaken for current status.
 */
export function KeptBriefing() {
  return (
    <Suspense fallback={null}>
      <KeptInner />
    </Suspense>
  );
}

function telHref(phone: string) {
  return `tel:+${phone.replace(/\D/g, "").replace(/^(?!1)/, "1")}`;
}

function KeptInner() {
  const id = useSearchParams().get("id");
  const [list, setList] = useState<Kept[] | null>(null);
  const [offline, setOffline] = useState(false);
  useEffect(() => {
    setList(readKept());
    setOffline(!navigator.onLine);
  }, []);
  const k = list ? (list.find((x) => x.id === id) ?? list[0] ?? null) : null;
  const download = (x: Kept) => {
    const url = URL.createObjectURL(new Blob([keptHtml(x)], { type: "text/html" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = keptFilename(x);
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const current = k ? coastHref({ place: k.snapshot.place.key, for: parseAudience(k.snapshot.audience), species: k.snapshot.species.id }) : "/coast";
  return (
    <InfoPage
      id="kept"
      testId="kept-page"
      kicker={`${KEPT_COPY.kicker}${offline ? KEPT_COPY.offline : ""}`}
      title={k ? keptTitle(k) : "Kept briefings"}
      titleProps={k ? { "data-testid": "kept-title" } : undefined}
      head={
        <>
          {k ? (
            <p className={i.small} data-testid="kept-saved">
              Saved {stamp(k.keptAt)}
              {k.snapshot.place.county ? ` · ${k.snapshot.place.county} County` : ""}
            </p>
          ) : null}
          <p className={i.small}>
            <Link href={current} className={i.link}>
              {offline ? KEPT_COPY.currentOffline : KEPT_COPY.current}
            </Link>
          </p>
        </>
      }
    >
      {list === null ? null : !k ? (
        <p className={i.note} data-testid="kept-empty">
          {KEPT_COPY.empty}
        </p>
      ) : (
        <div data-testid="kept-article">
          <p className={s.warning} data-testid="kept-warning">
            {KEPT_WARNING}
          </p>
          <div className={s.actions}>
            <Button variant="secondary" onClick={() => window.print()} data-testid="kept-print">
              {KEPT_COPY.print}
            </Button>
            <Button variant="secondary" onClick={() => download(k)} data-testid="kept-file">
              {KEPT_COPY.file}
            </Button>
            <Button
              variant="quiet"
              onClick={() => {
                deleteKept(k.id);
                setList(readKept());
              }}
              data-testid="kept-delete"
            >
              {KEPT_COPY.delete}
            </Button>
          </div>
          {keptSections(k).map((sec) => (
            <section key={sec.heading} className={s.section}>
              <h2>{sec.heading}</h2>
              <ul>
                {sec.lines.map((l, n) => {
                  const phone = /^(.*): (\(?\d[\d ()-]{8,})$/.exec(l.text);
                  return (
                    <li key={n}>
                      {phone ? (
                        <span className={s.text}>
                          {phone[1]}: <a href={telHref(phone[2])}>{phone[2]}</a>
                        </span>
                      ) : (
                        <span className={s.text}>{l.text}</span>
                      )}
                      {l.note ? <span className={s.lineNote}>{l.note}</span> : null}
                      {l.url ? (
                        <a href={l.url} target="_blank" rel="noopener noreferrer" className={s.url}>
                          {l.url}
                        </a>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
          <section className={s.agencies} data-testid="kept-agencies">
            <p className={s.agTitle}>{KEPT_COPY.agencies}</p>
            <p>{KEPT_COPY.agenciesNote}</p>
            <p className={s.tels}>
              <a href={`tel:${AGENCY_LINKS.cdfwClosureLine.tel}`} data-testid="kept-tel-cdfw">
                CDFW {AGENCY_LINKS.cdfwClosureLine.phone}
              </a>
              <a href={`tel:${AGENCY_LINKS.cdphLine.tel}`} data-testid="kept-tel-cdph">
                CDPH {AGENCY_LINKS.cdphLine.phone}
              </a>
            </p>
          </section>
          <p className={i.small}>
            {KEPT_COPY.builtFrom} {stamp(k.snapshot.dataGeneratedAt)}. {KEPT_COPY.notAgency}
          </p>
        </div>
      )}
      {list && list.length > 1 ? (
        <nav className={s.others} aria-label="Other kept briefings">
          <h2>{KEPT_COPY.others}</h2>
          <ul>
            {list
              .filter((x) => x.id !== k?.id)
              .map((x) => (
                <li key={x.id}>
                  <Link href={`/coast/kept?id=${encodeURIComponent(x.id)}`} className={i.link}>
                    {keptTitle(x)} · saved {stamp(x.keptAt)}
                  </Link>
                </li>
              ))}
          </ul>
        </nav>
      ) : null}
    </InfoPage>
  );
}
