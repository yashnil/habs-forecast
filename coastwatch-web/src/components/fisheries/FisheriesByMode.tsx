"use client";

import Link from "next/link";
import { useState } from "react";
import type { OfficialDataset } from "@/generated/official";
import { SPECIES, recordNamesSpecies, speciesPhrase, type SpeciesId } from "@/content/species";
import { coastHref } from "@/lib/coast";
import { ACTION_LABEL, currentRecords, disputedRecords } from "@/lib/official";
import { useMode } from "@/lib/mode";
import { formatDate } from "@/lib/time";
import { Mark } from "@/components/ui/Mark";

const ext = { target: "_blank", rel: "noopener noreferrer" } as const;
const link = "text-accent underline decoration-accent/35 underline-offset-2 hover:decoration-accent";

/**
 * Fisheries in Coastal mode: first, what applies to the species someone fishes now (official
 * notices that name it, the agency's statements and rules, and a way to plan a trip for their
 * waters); the statewide history of landed value follows, folded, labelled as history. In
 * Research mode the history leads, as before.
 */
export function FisheriesByMode({ official, children }: { official: OfficialDataset | null; children: React.ReactNode }) {
  const [mode, , ready] = useMode();
  const [sp, setSp] = useState<SpeciesId>("dungeness");
  // the server renders the Coastal layout (it works without JavaScript: the history is a native
  // <details>); a visitor set up for Research gets the history first once their choice is read
  const research = ready && mode === "research";
  const s = SPECIES.find((x) => x.id === sp)!;
  const notices = currentRecords(official).filter((r) => recordNamesSpecies(r.species, s));
  const disputed = disputedRecords(official).filter((d) => recordNamesSpecies(d.record.species, s));
  const statements = s.statementTopic ? (official?.registry.statements ?? []).filter((x) => x.topic.toLowerCase().includes(s.statementTopic!)) : [];
  return (
    <div className={research ? "" : "mx-auto w-full max-w-[1280px] space-y-6 px-4 pb-16 pt-5 sm:px-8"} data-testid={research ? "fisheries-research" : "fisheries-coastal"}>
      {!research && (
      <>
      <header className="space-y-2">
        <p className="text-[13px] font-medium text-accent">Fisheries</p>
        <h1 className="font-display text-[32px] font-medium leading-tight text-ink sm:text-[40px]">Your fishery, now</h1>
        <p className="max-w-[62ch] text-[16px] text-ink-2">Official notices that name a species, what the agency says about it, and where its rules are. The history of landed value is below.</p>
      </header>
      <div role="group" aria-label="Species" className="-mx-4 flex gap-1.5 overflow-x-auto px-4 pb-1 sm:mx-0 sm:flex-wrap sm:px-0" data-testid="fish-species">
        {SPECIES.filter((x) => x.id !== "any").map((x) => (
          <button
            key={x.id}
            type="button"
            aria-pressed={x.id === sp}
            onClick={() => setSp(x.id)}
            className={`min-h-[40px] shrink-0 rounded-full border px-3.5 text-[14px] ${x.id === sp ? "border-navy-900 bg-navy-900 text-white" : "border-hairline-strong bg-surface text-ink-2 hover:border-ink-3"}`}
            data-testid={`fish-species-${x.id}`}
          >
            {x.label}
          </button>
        ))}
      </div>
      <section className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]" aria-label={s.label}>
        <div className="rounded-[18px] border border-hairline border-l-[3px] border-l-official-line bg-surface px-4 py-4 sm:px-5" data-testid="fish-notices">
          <p className="cw-kind text-official-ink"><Mark kind="official" />Official notices naming {speciesPhrase(s)} · California</p>
          {/* an absence is a statement about the list's limits, never set larger than a finding (signature P6) */}
          {notices.length ? (
            <h2 className="mt-1 font-display text-[22px] font-medium text-ink">{`${notices.length} active notice${notices.length === 1 ? "" : "s"}`}</h2>
          ) : (
            <h2 className="mt-1 text-[16px] font-medium text-ink" data-testid="fish-notices-none">
              No notice in CoastWatch&apos;s list names {speciesPhrase(s)}
            </h2>
          )}
          {!notices.length && <p className="mt-1 text-[14px] text-ink-2">That is not the same as safe or open: the list may be incomplete and is not verified. Check the agency.</p>}
          <ul className="mt-2 divide-y divide-hairline">
            {disputed.map(({ record, dispute }) => (
              <li key={record.id} className="py-2.5">
                <p className="text-[13px] font-medium text-warning">Reported lifted by {dispute.agency} · awaiting review</p>
                <p className="text-[15px] font-medium text-ink">{record.title}</p>
                <a href={dispute.url} {...ext} className={`${link} text-[13px]`}>{dispute.agency} {dispute.release} ↗</a>
              </li>
            ))}
            {notices.map((r) => {
              const src = r.sources.find((x) => x.url.startsWith("https://"));
              return (
                <li key={r.id} className="py-2.5" data-testid={`fish-notice-${r.id}`}>
                  <p className="text-[13px] font-medium text-official-ink">{r.agency} · {ACTION_LABEL[r.action]}</p>
                  <p className="text-[15px] font-medium leading-snug text-ink">{r.title}</p>
                  <p className="text-[13px] text-ink-3">
                    {r.area.description} · {r.effective_date ? `since ${formatDate(r.effective_date, { year: true })}` : "start date not published"}
                    {src && (
                      <>
                        {" · "}
                        <a href={src.url} {...ext} className={link}>{r.agency} source ↗</a>
                      </>
                    )}
                  </p>
                </li>
              );
            })}
          </ul>
        </div>
        <div className="space-y-4">
          {statements.length > 0 && (
            <div className="rounded-[18px] border border-hairline bg-surface px-4 py-4 sm:px-5" data-testid="fish-statements">
              <p className="text-[13px] font-medium text-ink-3">What the agency says</p>
              <ul className="mt-2 space-y-2 text-[15px] text-ink">
                {statements.map((x) => (
                  <li key={x.id}>
                    “{x.statement}” <span className="text-[13px] text-ink-3">{x.agency} · <a href={x.source.url} {...ext} className={link}>source ↗</a></span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="rounded-[18px] border border-hairline bg-surface px-4 py-4 sm:px-5">
            <p className="text-[13px] font-medium text-ink-3">Before you go</p>
            <ul className="mt-2 space-y-2 text-[15px]">
              {s.link && (
                <li>
                  <a href={s.link.url} {...ext} className={link}>{s.link.label} ↗</a>
                </li>
              )}
              <li>
                <Link href={coastHref({ species: s.id })} className={link} data-testid="fish-plan">
                  Plan a trip for your waters in My Coast →
                </Link>
              </li>
            </ul>
          </div>
        </div>
      </section>
      </>
      )}
      {/* the history is always inside this one element, so it is never re-mounted when the mode is read */}
      <details open={research ? true : undefined} className={research ? "contents" : "group rounded-[18px] border border-hairline bg-surface"} data-testid="fisheries-history">
        <summary className={research ? "hidden" : "flex min-h-[52px] cursor-pointer items-center justify-between gap-3 px-4 text-[15px] font-medium text-ink sm:px-5"}>
          <span>
            History: what California&apos;s toxin-affected fisheries have landed
            <span className="block text-[13px] font-normal text-ink-3">Statewide landed value 2015–2024 from NOAA. Past landings, not this season or these waters.</span>
          </span>
          <span aria-hidden className="text-ink-3 transition-transform group-open:rotate-90">›</span>
        </summary>
        <div className={research ? "" : "border-t border-hairline"}>{children}</div>
      </details>
    </div>
  );
}
