"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Button, dayTime, Glyph, Micro } from "@/nx/ds";
import { groupUnchecked } from "../feed";
import type { Recheck } from "../types";
import s from "./evidence.module.css";

/**
 * Recheck before departure (v15 §9): the one action and its four-part answer: what changed, what is the same, what
 * could not be checked, and the official pages to read directly. The time and completeness are always visible; an
 * incomplete recheck says so in its stamp and never reads as a pass. Offline, the stamp says it compared the copy
 * saved on this device.
 */
export type RecheckPanelProps = {
  recheck: Recheck | null;
  checking: boolean;
  onRecheck?: () => void;
  now: string;
  /** the workspace's one filled button may be elsewhere */
  actionVariant?: "primary" | "secondary";
  /** the original time of a saved copy (RecheckResult.snapshot.savedFrom) */
  savedFrom?: string | null;
};

export function RecheckPanel({ recheck: r, checking, onRecheck, now, actionVariant = "primary", savedFrom = null }: RecheckPanelProps) {
  const reduce = useReducedMotion();
  const stamp = !r
    ? "Not rechecked yet"
    : r.mode === "saved"
      ? `Offline · compared against the copy saved on this device${savedFrom ? ` ${dayTime(savedFrom, { ref: now })}` : ""} · incomplete`
      : `Checked ${dayTime(r.at, { ref: now })} · ${r.complete ? "complete" : "incomplete"}`;
  return (
    <section className={s.panel} aria-label="Recheck" data-testid="trip-recheck-panel" data-complete={r?.complete || undefined} data-mode={r?.mode}>
      <div className={s.panelHead}>
        <div>
          <Micro>Recheck before departure</Micro>
          <p className={s.stamp} data-complete={r?.complete || undefined} data-testid="trip-recheck-stamp">
            {stamp}
          </p>
        </div>
        {onRecheck ? (
          <Button variant={actionVariant} onClick={onRecheck} busy={checking} busyLabel="Checking…" data-testid="trip-recheck-action">
            {r ? "Recheck again" : "Recheck sources"}
          </Button>
        ) : null}
      </div>
      <AnimatePresence initial={false}>
        {r ? (
          <motion.div key={r.snapshotId} className={s.parts} initial={reduce ? false : { y: 4 }} animate={{ y: 0 }} transition={reduce ? { duration: 0 } : { duration: 0.16, ease: [0.2, 0, 0, 1] }}>
            <Part id="changed" label="Changed" n={r.changed.length}>
              {r.changed.length ? (
                <ul className={s.partList}>
                  {r.changed.map((c) => (
                    <li key={c.id}>{c.title}</li>
                  ))}
                </ul>
              ) : (
                <p className={s.empty}>{r.complete ? "No changes found in the sources checked." : "None found in what could be checked."}</p>
              )}
            </Part>
            <Part id="same" label={r.mode === "saved" ? "Same in the saved copy" : "Same as planned"} n={r.same.length}>
              {r.same.length ? (
                <ul className={s.partList}>
                  {r.same.map((x) => (
                    <li key={x.key}>{x.title}</li>
                  ))}
                </ul>
              ) : (
                <p className={s.empty}>Nothing compared unchanged.</p>
              )}
            </Part>
            <Part id="unchecked" label="Couldn't check" n={r.unchecked.length}>
              {r.unchecked.length ? (
                <ul className={s.partList}>
                  {groupUnchecked(r.unchecked).map((g) => (
                    <li key={g.sources.join()}>
                      {g.labels}
                      <small>{g.why}</small>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className={s.empty}>Every source answered.</p>
              )}
            </Part>
            <Part id="stale" label="Answered, but old" n={r.stale.length}>
              {r.stale.length ? (
                <ul className={s.partList}>
                  {r.stale.map((x) => (
                    <li key={x.key}>
                      {x.title}
                      <small>{x.asOf ? `Dated ${dayTime(x.asOf, { ref: now })}; it says nothing about today.` : "Undated."}</small>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className={s.empty}>No old records among those that answered.</p>
              )}
            </Part>
            <Part id="verify" label="Verify yourself" n={r.verify.length}>
              {r.verify.length ? (
                <ul className={s.partList}>
                  {r.verify.map((x) => (
                    <li key={x.url}>
                      <a className={s.link} href={x.url} target="_blank" rel="noreferrer noopener">
                        {x.label} ↗
                      </a>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className={s.empty}>No official page on record for this trip.</p>
              )}
              <p className={s.empty}>
                <Glyph kind="unverified-coral" /> Official notices are not verified by a person yet.
              </p>
            </Part>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </section>
  );
}

function Part({ id, label, n, children }: { id: string; label: string; n: number; children: React.ReactNode }) {
  return (
    <div className={s.part} data-part={id} data-some={n > 0 || undefined} data-testid={`trip-recheck-${id}`}>
      <div className={s.partHead}>
        <Micro as="h3">{label}</Micro>
        <span className={s.n}>{n}</span>
      </div>
      {children}
    </div>
  );
}
