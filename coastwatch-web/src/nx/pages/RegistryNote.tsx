"use client";

import Link from "next/link";
import { useEffect, useId, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { OFFICIAL_STATUS } from "@/content/copy";
import { REGISTRY_LINE } from "@/content/pages";
import type { VerificationState } from "@/lib/official";
import { panelIn } from "@/nx/ds/motion";
import { Icon } from "@/nx/ds/Icon";
import { Glyph } from "@/nx/ds/synth/Grammar";
import s from "./registry.module.css";

/** The registry, as the server read it (registry.server.ts). */
export type RegistryTrust = {
  fixture: boolean;
  generatedAt: string | null;
  state: VerificationState;
  reasons: string[];
  /** current notices, or null when Neritic's copy could not be read */
  records: Array<{ id: string; title: string; line: string }> | null;
};

const CDFW = OFFICIAL_STATUS.links[0];
const CDPH = OFFICIAL_STATUS.links[1];

/**
 * The registry line (v15.1): what the v4 trust pill said on the legacy pages, as one quiet line under an informational
 * page's head. In words at every width: research preview, and the official notices not verified (the registry says
 * "verified" only once a person has reviewed it, cwp review-official). It opens in place: the full statement, the two
 * agencies, each current notice by name (all of them, with wording and dates, on /notices) and the hotlines.
 * Test ids kept from the pill: notices-button, registry-disclosure, notices-panel, fixture-banner.
 */
export function RegistryNote({ trust }: { trust: RegistryTrust | null }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const state: VerificationState = trust?.state ?? "unavailable";
  const word = OFFICIAL_STATUS.verification[state];
  const records = trust?.records ?? null;
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setOpen(false);
      button.current?.focus();
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [open]);
  return (
    <div className={s.reg}>
      {/* design-lint-allow raw-button: the registry line is a disclosure row whose button keeps the trust pill's test ids */}
      <button ref={button} type="button" className={s.line} aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)} data-testid="notices-button" data-state={state}>
        <Glyph kind="unverified-coral" className={s.glyph} />
        <span className={s.words} data-testid="registry-disclosure">
          {trust?.fixture ? (
            <>
              <span className={s.fixture} role="status" data-testid="fixture-banner">
                {REGISTRY_LINE.fixture}
                <span className={s.sr}>{REGISTRY_LINE.fixtureSr}</span>
              </span>
              <span className={s.sep} aria-hidden="true">
                ·
              </span>
            </>
          ) : null}
          <span data-testid="preview-disclosure">{REGISTRY_LINE.preview}</span>
          <span className={s.sep} aria-hidden="true">
            ·
          </span>
          <span className={s.count}>
            {records ? `${records.length} official ${records.length === 1 ? "notice" : "notices"} in CA` : "Official notices"}
          </span>
          <span className={s.sep} aria-hidden="true">
            ·
          </span>
          <b className={s.state}>{word}</b>
        </span>
        <Icon name="chevron" size={16} direction={open ? "up" : "down"} className={s.chev} />
      </button>
      <AnimatePresence initial={false}>
        {open ? (
          <motion.div id={id} role="region" aria-label="Official notices" className={s.panel} data-testid="notices-panel" initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={panelIn}>
            <p className={s.lead} data-testid="notices-unverified" data-state={state}>
              {state === "verified" ? "Checked by a person against the agencies' pages." : `${word}. ${OFFICIAL_STATUS.notVerifiedLead}`} {REGISTRY_LINE.checkAgencies}
            </p>
            <p className={s.text} data-testid="preview-statement">
              <Link href="/about" className={s.inline}>
                {REGISTRY_LINE.preview}
              </Link>
              {REGISTRY_LINE.previewStatement}
            </p>
            <p className={s.text}>{OFFICIAL_STATUS.notTracked}</p>
            <p className={s.agencies}>
              <a href={CDFW.href} target="_blank" rel="noreferrer" title={CDFW.label}>
                CDFW
              </a>
              <a href={CDPH.href} target="_blank" rel="noreferrer" title={CDPH.label}>
                CDPH
              </a>
            </p>
            {records ? (
              <ul className={s.list} data-testid="notices-current">
                {records.map((r) => (
                  <li key={r.id} data-testid={`notice-${r.id}`}>
                    <span className={s.recLine}>{r.line}</span>
                    <span className={s.recTitle}>{r.title}</span>
                    <span className={s.recState}>
                      <Glyph kind="unverified-coral" /> {word}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className={s.text}>{REGISTRY_LINE.noData}</p>
            )}
            <p className={s.more}>
              <Link href="/notices" className={s.inline}>
                {REGISTRY_LINE.allNotices}
              </Link>
            </p>
            <p className={s.missing} data-testid="missing-not-open">
              {OFFICIAL_STATUS.missingNotOpen}
            </p>
            <dl className={s.hotlines} data-testid="notices-hotlines">
              {OFFICIAL_STATUS.hotlines.map((h) => (
                <div key={h.tel}>
                  <dt>{h.label}</dt>
                  <dd>
                    <a href={`tel:${h.tel}`}>{h.phone}</a>
                  </dd>
                </div>
              ))}
            </dl>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
