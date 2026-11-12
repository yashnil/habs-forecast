"use client";

import { useState, type ReactNode } from "react";
import { Button } from "@/nx/ds/Button";
import { Disclosure } from "@/nx/ds/Controls";
import { dayTime } from "@/nx/ds/format";
import { markFor } from "@/nx/ds/kinds";
import { Mark } from "@/nx/ds/Mark";
import { ProvenanceSheet, Stamp, type Provenance } from "@/nx/ds/Stamp";
import { Loading } from "@/nx/ds/States";
import type { Gate, GateFact, GateId } from "./gates";
import s from "./gates.module.css";

/**
 * What decides the next trip, as three ruled parts in a fixed order (gates.ts): whale measures, domoic acid, the
 * sea. Each part says who decides or measures, the fact in one line with its date, how it relates to the other
 * parts (fixed sentences), and what to do next. Every line of fact opens where it comes from. Nothing here is a
 * verdict: the parts are evidence side by side, never combined into a score.
 */
export type GateExtra = { summary: string; items: Array<{ text: string; provenance?: Provenance | null }> };

export function Gates({ gates, now, slots, extras, portName }: { gates: Gate[]; now: string; portName: string; slots?: Partial<Record<GateId, ReactNode>>; extras?: Partial<Record<GateId, GateExtra>> }) {
  const [open, setOpen] = useState<Provenance | null>(null);
  return (
    <section className={s.gates} aria-label={`What decides a trip from ${portName}`} data-testid="nx-gates" data-tour="gates">
      {open ? <ProvenanceSheet provenance={open} now={now} open onOpenChange={(o) => !o && setOpen(null)} /> : null}
      {gates.map((g) => (
        <GatePart key={g.id} gate={g} now={now} slot={slots?.[g.id]} extra={extras?.[g.id]} onOpen={setOpen} />
      ))}
    </section>
  );
}

function GatePart({ gate: g, now, slot, extra, onOpen }: { gate: Gate; now: string; slot?: ReactNode; extra?: GateExtra; onOpen: (p: Provenance) => void }) {
  // the sea gate's headline is its first fact (an advisory when one is in force): the rest follow as lines
  const facts = g.id === "sea" && g.state === "ok" ? g.facts.slice(1) : g.facts;
  const lead = g.id === "sea" && g.state === "ok" ? g.facts[0] : null;
  return (
    <article className={s.gate} data-gate={g.id} data-state={g.state} data-testid={`nx-gate-${g.id}`} aria-labelledby={`nx-gate-${g.id}-title`}>
      <h2 className={s.title} id={`nx-gate-${g.id}-title`}>
        {g.title}
        <span className={s.by}> · {g.authority}</span>
      </h2>
      {g.state === "loading" ? (
        <Loading>{g.headline}</Loading>
      ) : (
        <p className={s.headline} data-role={lead?.role} data-testid={`nx-gate-${g.id}-headline`}>
          {g.headline}
          {g.state === "old" ? (
            <>
              {" "}
              <Mark kind="stale" />
            </>
          ) : null}
          {/* a sample is read with its age (T6, T17) */}
          {g.dateRole === "sampled" && g.age ? <span className={s.age}> · {g.age}</span> : null}
        </p>
      )}
      {g.provenance ? (
        <p className={s.stamp}>
          <Stamp provenance={g.provenance} now={now} />
        </p>
      ) : null}
      {slot}
      {facts.length ? (
        <ul className={s.facts}>
          {facts.map((f) => (
            <li key={f.id}>
              <FactLine fact={f} now={now} onOpen={onOpen} />
            </li>
          ))}
        </ul>
      ) : null}
      {g.relation.map((r) => (
        <p key={r} className={s.relation} data-testid={`nx-gate-${g.id}-relation`}>
          {r}
        </p>
      ))}
      {extra && extra.items.length ? (
        <Disclosure summary={extra.summary} className={s.extra} data-testid={`nx-gate-${g.id}-more`}>
          <ul className={s.extraList}>
            {extra.items.map((d, i) => (
              <li key={i}>
                {d.text ? <p>{d.text}</p> : null}
                {d.provenance ? <Stamp provenance={d.provenance} now={now} /> : null}
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : null}
      {g.actions.length ? (
        <ul className={s.actions} aria-label={`Next steps: ${g.title.toLowerCase()}`}>
          {g.actions.map((a) => (
            <li key={a.id}>
              <Button variant="text" href={a.href} external={a.external} data-testid={`nx-gate-${g.id}-${a.id}`}>
                {a.label}
              </Button>
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}

/** One line of fact: its kind's mark, the words, when it is from. The whole line opens where it comes from. */
function FactLine({ fact: f, now, onOpen }: { fact: GateFact; now: string; onOpen: (p: Provenance) => void }) {
  const p = f.provenance;
  const when = p ? `${p.source}, ${p.dateRole && p.dateRole !== "measured" ? `${p.dateRole} ` : ""}${dayTime(p.date, { ref: now })}` : null;
  const body = (
    <>
      {p ? <Mark kind={markFor(p.kind)} withWords={false} className={s.mark} /> : <span className={s.mark} aria-hidden="true" />}
      <span className={s.factText}>
        {f.text}
        {f.old ? (
          <>
            {" "}
            <Mark kind="stale" />
          </>
        ) : null}
        {when ? <span className={s.when}>{when}</span> : null}
      </span>
    </>
  );
  if (!p) return <span className={s.fact}>{body}</span>;
  return (
    <button type="button" className={s.fact} data-role={f.role} onClick={() => onOpen(p)} aria-haspopup="dialog" aria-label={`${f.text}${f.old ? ", old" : ""}. ${when}. Open where this comes from`}>
      {body}
    </button>
  );
}

/** NWS advisories in force, set right under the status: the first thing a returning reader must not miss. */
export function Advisories({ items, now }: { items: GateFact[]; now: string }) {
  if (!items.length) return null;
  return (
    <ul className={s.advisories} aria-label="NWS advisories in force" data-testid="nx-advisories">
      {items.map((a) => (
        <li key={a.id}>
          <p className={s.advisory}>{a.text}</p>
          {a.provenance ? <Stamp provenance={a.provenance} now={now} /> : null}
        </li>
      ))}
    </ul>
  );
}
