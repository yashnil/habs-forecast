"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { ChartRegulation, noticeExtents } from "@/nx/chart/regulation";
import { ChartScene } from "@/nx/chart/scene";
import { Button } from "@/nx/ds/Button";
import { day } from "@/nx/ds/format";
import { Legend } from "@/nx/ds/Mark";
import { Stamp, type Provenance } from "@/nx/ds/Stamp";
import { SourcesDiffer, Unavailable } from "@/nx/ds/States";
import { Statement } from "@/nx/ds/Statement";
import { savedPortKey, useCurrentPort } from "@/nx/shell/port";
import { COAST_VIEW, portScene } from "@/nx/shell/views";
import type { NoticeItem, NoticesView } from "./model";
import s from "./notices.module.css";

const ymd = (iso: string) => iso.slice(0, 10);

/** When it took effect and when it is expected to end, in the record's own dates; an end is never confirmed here. */
function datesLine(n: NoticeItem, now: string): string {
  const today = ymd(now);
  const parts: string[] = [];
  if (n.effective) parts.push(n.effective <= today ? `In effect since ${day(n.effective, { ref: now, year: true })}.` : `Takes effect ${day(n.effective, { ref: now, year: true })}.`);
  else parts.push(n.effectiveNote ? `${n.effectiveNote.replace(/\.$/, "")}.` : "The notice does not state an effective date.");
  if (n.expectedEnd && n.expectedEnd >= today) parts.push(`Expected to end ${day(n.expectedEnd, { ref: now, year: true })}; not an end until ${n.agency} announces it.`);
  return parts.join(" ");
}

function provenance(n: NoticeItem, v: NoticesView): Provenance {
  return {
    kind: "official",
    title: n.title,
    value: `${n.agency}, ${n.action.toLowerCase()}`,
    source: n.agency,
    sourceLong: n.sources[0]?.label ?? n.agency,
    sourceUrl: n.sources[0]?.url,
    date: v.review.reviewedAt ?? n.effective ?? v.review.checkedAt ?? "",
    dateRole: "checked",
    place: n.area,
    method: v.review.method ?? undefined,
    caveats: [...n.flags, ...n.caveats],
  };
}

function Entry({ n, v, now, full }: { n: NoticeItem; v: NoticesView; now: string; full: boolean }) {
  return (
    <li className={s.entry} id={`notice-${n.id}`} data-testid="nx-notice" data-notice={n.id}>
      <p className={s.meta}>
        {n.agency} · {n.action} · {n.fishery}
        {n.statewide ? " · statewide" : ""}
      </p>
      <h3 className={s.title}>{n.title}</h3>
      {full ? <p className={s.summary}>{n.summary}</p> : null}
      <p className={s.dates}>{datesLine(n, now)}</p>
      {n.flags.map((f) => (
        <p key={f} className={s.flag}>
          {f}
        </p>
      ))}
      <p className={s.area}>
        {n.area}
        {full && n.basis ? ` ${n.basis}` : ""}
      </p>
      {full
        ? n.caveats.map((c) => (
            <p key={c} className={s.area}>
              {c}
            </p>
          ))
        : null}
      {(() => {
        const p = provenance(n, v);
        return p.date ? <Stamp provenance={p} now={now} /> : null;
      })()}
      {n.sources.length ? (
        <ul className={s.sources} aria-label={`Sources for ${n.title}`}>
          {n.sources.map((src) => (
            <li key={src.url}>
              <a href={src.url} target="_blank" rel="noreferrer" className={s.source}>
                {src.label}
              </a>
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function Group({ id, title, lead, items, v, now, full }: { id: string; title: string; lead?: string; items: NoticeItem[]; v: NoticesView; now: string; full: boolean }) {
  if (!items.length) return null;
  return (
    <section className={s.group} aria-labelledby={`nx-notices-${id}`} data-testid={`nx-notices-${id}`}>
      <h2 id={`nx-notices-${id}`} className={s.head}>
        {title} <span className={s.count}>{items.length}</span>
      </h2>
      {lead ? <p className={s.lead}>{lead}</p> : null}
      <ul className={s.list}>
        {items.map((n) => (
          <Entry key={n.id} n={n} v={v} now={now} full={full} />
        ))}
      </ul>
    </section>
  );
}

/**
 * Every current official notice for the reader's port and the rest of the coast (reached from Today's notices
 * row and from Alerts). Each notice: agency, action, dates, area, source links and "not verified" on its stamp.
 * A record an agency release appears to lift is shown as "sources differ", quoting both, never as lifted.
 */
export function NoticesPage({ view, now }: { view: NoticesView | null; now: string }) {
  const router = useRouter();
  const current = useCurrentPort();
  // an address without a port opens the reader's saved port, when this device has one
  useEffect(() => {
    if (view && !view.port) {
      const k = savedPortKey();
      if (k) router.replace(`/notices?place=${encodeURIComponent(k)}`);
    }
  }, [view, router]);
  const port = view?.port ? (current.port?.key === view.port.key ? current.port : null) : null;
  const scene = port ? portScene("notices", port, 60) : { id: "notices:coast", view: COAST_VIEW };

  if (!view) {
    return (
      <section className={s.page} data-testid="nx-notices">
        <ChartScene scene={scene} />
        <p className={s.kicker}>Official notices</p>
        <Unavailable
          what="Official notices aren't available"
          why="Neritic's registry of CDFW and CDPH notices did not load. No notice listed here does not mean open."
          next={
            <Button variant="text" href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" external>
              CDFW health advisories
            </Button>
          }
        />
      </section>
    );
  }
  const name = view.port?.name ?? null;
  const here = view.here.length + view.unsettled.length;
  // on the chart: the latitude bands of the notices listed for the port (all of them with no port), in magenta
  const extents = noticeExtents(view.port ? [...view.here, ...view.unsettled] : [...view.here, ...view.unsettled, ...view.elsewhere]);
  const reviewWords =
    view.review.status === "human_verified"
      ? `The registry records a review against the agencies' pages${view.review.reviewedAt ? ` on ${day(view.review.reviewedAt, { ref: now })}` : ""}; Neritic still does not stand in for them.`
      : `The registry is transcribed from the agencies' pages${view.review.reviewedAt ? ` (${day(view.review.reviewedAt, { ref: now })})` : ""} and awaits a person's review, so every notice here is not verified.`;
  return (
    <section className={s.page} data-testid="nx-notices">
      <ChartScene scene={scene} />
      <ChartRegulation spec={extents.length ? { extents } : null} />
      <Statement
        size="md"
        as="h1"
        scope={<span className={s.scope}>{name ? `${name} · official notices` : "Official notices"}</span>}
        title={name ? `${here} ${here === 1 ? "notice applies" : "notices apply"} at ${name}` : `${view.total} current notices on the coast`}
        why={name ? `And ${view.elsewhere.length} elsewhere on the California coast.` : null}
      >
        <p className={s.lead}>
          CDFW and CDPH notices as Neritic&rsquo;s registry lists them. {reviewWords} No notice listed does not mean open; the agencies are the authority.
        </p>
      </Statement>

      {view.differ.length ? (
        <section className={s.group} aria-labelledby="nx-notices-differ" data-testid="nx-notices-differ">
          <h2 id="nx-notices-differ" className={s.head}>
            Sources differ <span className={s.count}>{view.differ.length}</span>
          </h2>
          <p className={s.lead}>An agency release appears to lift these. Until a person reviews it, Neritic shows each as neither in effect nor lifted.</p>
          <ul className={s.list}>
            {view.differ.map((n) => (
              <li key={n.id} className={s.entry} data-testid="nx-notice" data-notice={n.id}>
                <SourcesDiffer
                  subject={`${n.title} (${n.area})`}
                  claims={[
                    { agency: n.agency, quote: n.title, reference: "Neritic's registry lists it as active", url: n.sources[0]?.url ?? "https://www.cdph.ca.gov" },
                    { agency: n.dispute!.agency, quote: n.dispute!.title, reference: `${n.dispute!.agency} release ${n.dispute!.release}`, url: n.dispute!.url },
                  ]}
                  checked={view.review.checkedAt ?? view.review.reviewedAt ?? now}
                  now={now}
                  next={<>Check {n.agency} before you act.</>}
                />
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <Group id="here" title={name ? `At ${name}` : "In effect"} lead={name ? "Notices whose official area covers this port's waters, statewide ones included." : undefined} items={name ? view.here : [...view.here, ...view.elsewhere]} v={view} now={now} full />
      <Group id="unsettled" title={`Reach not settled at ${name ?? "this port"}`} lead="The registry does not settle whether these reach this port's waters. Read the notice." items={view.unsettled} v={view} now={now} full />
      {name ? <Group id="elsewhere" title="Elsewhere on the coast" items={view.elsewhere} v={view} now={now} full={false} /> : null}
      {view.conflicts.length ? (
        <section className={s.group} aria-labelledby="nx-notices-conflicts">
          <h2 id="nx-notices-conflicts" className={s.head}>
            Records that disagree
          </h2>
          {view.conflicts.map((c) => (
            <p key={c} className={s.lead}>
              {c}
            </p>
          ))}
        </section>
      ) : null}
      <Legend className={s.legend} kinds={["official"]} title="Marks on this page" />
      {view.port ? (
        <p className={s.back}>
          <Button variant="text" href={`/today?place=${encodeURIComponent(view.port.key)}`}>
            Back to Today at {view.port.name}
          </Button>
        </p>
      ) : null}
    </section>
  );
}
