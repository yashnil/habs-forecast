"use client";

import Link from "next/link";
import { MotionConfig } from "motion/react";
import { useEffect, useId, useRef, useState } from "react";
import { ChartScene } from "@/nx/chart/scene";
import { Button, Hint, VisuallyHidden } from "@/nx/ds";
import { Desk, DeskAside, DeskBody, DeskFoot, DeskHeader, DeskMain, PageLayout } from "@/nx/ds/layout";
import { ChartToggle } from "@/nx/ds/layout/ChartToggle";
import { savedPortKey, usePorts } from "@/nx/shell/port";
import { portScene } from "@/nx/shell/views";
import { PILOT_PORTS } from "@/pilot/contracts";
import { READINESS_WORDS } from "@/pilot/readiness";
import { segments } from "@/pilot/ask/sms";
import { suggestionsFor } from "@/pilot/ask/suggestions";
import { DEMO_LABEL, DISCLAIMER, LIVE_LABEL, SAMPLE_DATA_LABEL } from "@/pilot/ask/templates";
import type { AskAnswer, AskMeta, Citation, SeenSnapshot } from "@/pilot/ask/types";
import { UNVERIFIED_LINE } from "@/pilot/ask/unverified";
import { savedCoverage, uncoveredWords, type AskPort } from "@/pilot/ask/ui/coverage";
import { readSeen, writeSeen } from "@/pilot/ask/ui/seen";
import s from "./ask.module.css";

/**
 * Ask Neritic in the v13 shell: a desk page whose work is the conversation, with the chart showing the port
 * the answers are about. Answers come only from the records on that port's Today page (/api/ask), each line
 * with its source and date as a chip that opens the source and the Today row.
 *
 * Ask covers the pilot ports only. A reader who arrives from any other port is told so, is linked to Today at
 * their own port, and sees no question box until they choose a pilot port: Ask never answers about one port
 * as though it were another. The text-message preview is shown only where text messages are in their demo
 * state; where they are not connected at all, nothing about texts is shown.
 */
type Turn = { q: string; a: AskAnswer | null; error?: string };

const BASIS: Record<Citation["basis"], string> = {
  official: "Official record",
  measured: "Measurement",
  forecast: "Forecast",
  outlook: "Outlook",
  analysis: "Analysis",
  derived: "Neritic",
};

export type AskPageProps = {
  /** the pilot port to open on, or "" when the port asked for is outside the pilot */
  initialPlace: string;
  uncovered: AskPort | null;
  /** no port was named in the address: the reader's saved port decides */
  preferSaved: boolean;
  meta: AskMeta;
  suggestions: string[];
};

export function AskPage({ initialPlace, uncovered, preferSaved, meta, suggestions: sugProp }: AskPageProps) {
  const ports = usePorts();
  const [place, setPlace] = useState(uncovered ? "" : initialPlace);
  const [outside, setOutside] = useState<AskPort | null>(uncovered);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const prevSeen = useRef<SeenSnapshot | null>(null);
  const inputId = useId();
  const endRef = useRef<HTMLDivElement>(null);
  const label = PILOT_PORTS.find((p) => p.place_id === place)?.label ?? "a pilot port";
  const waiting = !place;
  const suggestions = waiting ? [] : place === initialPlace && sugProp.length ? sugProp : suggestionsFor(place);
  const shown = ports.find((p) => p.key === (place || outside?.key)) ?? null;

  useEffect(() => {
    prevSeen.current = place ? readSeen(place) : null;
  }, [place]);

  useEffect(() => {
    if (!preferSaved) return;
    // the reader's own port: a pilot port opens; any other is named as not covered, never swapped for a pilot port
    const names = Object.fromEntries(ports.map((p) => [p.key, p.name]));
    const c = savedCoverage(savedPortKey(), names);
    if (c?.state === "covered") setPlace(c.place);
    else if (c?.state === "uncovered") {
      setOutside(c.port);
      setPlace("");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preferSaved]);

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy || !place) return;
    setBusy(true);
    setQ("");
    const first = turns.length === 0;
    setTurns((t) => [...t, { q: text, a: null }]);
    try {
      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text, place, first, since: prevSeen.current }) });
      const body = (await res.json()) as AskAnswer & { error?: string };
      if (!res.ok) throw new Error(body.error ?? "Ask Neritic is unavailable right now.");
      if (body.snapshot && body.place) writeSeen(body.place.place_id, body.snapshot);
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, a: body } : x)));
    } catch (e) {
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error: e instanceof Error ? e.message : "Something went wrong." } : x)));
    } finally {
      setBusy(false);
      requestAnimationFrame(() => endRef.current?.scrollIntoView({ block: "nearest" }));
    }
  }

  const last = [...turns].reverse().find((t) => t.a)?.a ?? null;
  const live = meta.readiness.ask === "live";
  const words = outside ? uncoveredWords(outside) : null;
  // a preview of the text a phone would get is only honest where texts exist in some form
  const smsDemo = meta.readiness.sms === "demo";

  return (
    <MotionConfig reducedMotion="user">
      <ChartScene scene={portScene("ask", shown, 30)} />
      <PageLayout mode="desk" paneKey="ask" defaultPane="rail" />
      <article className={s.page} data-testid="ask-chat">
        <Desk>
          <DeskHeader
            context="Ask Neritic · private pilot"
            title="Ask Neritic"
            status={waiting ? "A plain question about a pilot port, answered only from the records on its Today page." : `A plain question about ${label}, answered only from the records on its Today page. Every line names its source and its date.`}
            actions={<ChartToggle />}
            pinned={
              <div className={s.pinned}>
                <label htmlFor={`${inputId}-port`} className={s.label}>
                  Port
                </label>
                <select
                  id={`${inputId}-port`}
                  className={s.select}
                  value={place}
                  onChange={(e) => {
                    if (!e.target.value) return;
                    setPlace(e.target.value);
                    setOutside(null);
                  }}
                  data-testid="ask-port"
                >
                  {waiting ? <option value="">Choose a pilot port</option> : null}
                  {PILOT_PORTS.map((p) => (
                    <option key={p.place_id} value={p.place_id}>
                      {p.label}
                    </option>
                  ))}
                </select>
                {meta.dataset === "fixture" ? (
                  <span className={s.tag} data-testid="ask-sample">
                    Sample data
                  </span>
                ) : null}
              </div>
            }
          >
            <p className={s.readiness} data-testid="ask-readiness">
              <span className={s.tag} data-readiness={meta.readiness.ask}>
                {READINESS_WORDS[meta.readiness.ask]}
              </span>
              <span>{live ? LIVE_LABEL : DEMO_LABEL}</span>
            </p>
          </DeskHeader>
          <DeskBody layout={smsDemo ? "main-aside" : "main-full"} aside="end">
            <DeskMain>
              {outside && words ? (
                <Hint tone="caution">
                  <span data-testid="ask-uncovered" data-port={outside.key}>
                    <strong>{words.title}</strong> {words.body}{" "}
                    <Link href={`/today?place=${encodeURIComponent(outside.key)}`} className={s.link} data-testid="ask-uncovered-today">
                      Today at {outside.name ?? "your port"}
                    </Link>{" "}
                    has its season status, tests and notices, each with its source.
                  </span>
                </Hint>
              ) : null}
              {turns.length === 0 && !waiting ? (
                <>
                  <p className={s.note} data-testid="ask-disclaimer">
                    {DISCLAIMER}
                  </p>
                  {meta.logging ? <p className={s.small}>Questions are logged for 90 days to check answers, then deleted.</p> : null}
                </>
              ) : null}

              <ol className={s.turns} aria-live="polite" aria-busy={busy} data-testid="ask-turns">
                {turns.map((t, i) => (
                  <li key={i}>
                    <p className={s.asked}>You asked</p>
                    <p className={s.question} data-testid="ask-question">
                      {t.q}
                    </p>
                    {t.a ? (
                      <Answer a={t.a} />
                    ) : t.error ? (
                      <p role="alert" className={s.error} data-testid="ask-error">
                        {t.error}
                      </p>
                    ) : (
                      <p className={s.small}>Reading the records for {label}</p>
                    )}
                  </li>
                ))}
              </ol>
              <div ref={endRef} />

              {waiting ? null : (
                <>
                  <h2 className={s.h2} id={`${inputId}-sug`}>
                    Common questions for {label}
                  </h2>
                  <ul className={s.suggestions} aria-labelledby={`${inputId}-sug`} data-testid="ask-suggestions">
                    {suggestions.map((x) => (
                      <li key={x}>
                        <button type="button" disabled={busy} onClick={() => void ask(x)} className={s.suggest}>
                          {x}
                        </button>
                      </li>
                    ))}
                  </ul>
                  <form
                    className={s.form}
                    onSubmit={(e) => {
                      e.preventDefault();
                      void ask(q);
                    }}
                  >
                    <VisuallyHidden>
                      <label htmlFor={inputId}>Ask about {label}</label>
                    </VisuallyHidden>
                    <input id={inputId} value={q} onChange={(e) => setQ(e.target.value)} maxLength={400} placeholder={`Ask about ${label}`} autoComplete="off" className={s.input} data-testid="ask-input" />
                    <Button variant="primary" type="submit" disabled={busy || !q.trim()} data-testid="ask-send">
                      Ask
                    </Button>
                  </form>
                </>
              )}
            </DeskMain>
            {smsDemo ? (
              <DeskAside label="Text message preview" data-testid="ask-sms-preview">
                <h2 className={s.h2}>Text message preview</h2>
                <p className={s.small} data-testid="ask-sms-label">
                  Text messages are in carrier review. This is a preview only: nothing is sent.
                </p>
                {last?.sms ? (
                  <>
                    <p className={s.sms} data-testid="ask-sms-text">
                      {last.sms}
                    </p>
                    <p className={s.small}>
                      {last.sms.length} characters · {segments(last.sms)} SMS segment{segments(last.sms) === 1 ? "" : "s"}
                    </p>
                  </>
                ) : (
                  <p className={s.small}>Ask a question to see the exact reply a phone would get.</p>
                )}
              </DeskAside>
            ) : null}
          </DeskBody>
          <DeskFoot>
            <p>
              Your question is sent to Neritic to be answered. Nothing else on this device is sent with it.{" "}
              {waiting ? null : (
                <Link href={`/today?place=${place}`} className={s.link} data-testid="ask-today-link">
                  Today at {label}
                </Link>
              )}
            </p>
          </DeskFoot>
        </Desk>
      </article>
    </MotionConfig>
  );
}

function Chip({ c, i }: { c: Citation; i: number }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  // the date as Today words it ("Oct 12"); an answer from an older server carries only the day
  const date = c.as_of_words ?? c.as_of?.slice(0, 10) ?? "no date";
  return (
    <li>
      <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen((v) => !v)} className={s.chip} data-testid="ask-chip">
        <span className={s.chipNo} aria-hidden="true">
          {i + 1}
        </span>
        <span className={s.chipName}>{c.source_name}</span>
        <span className={s.chipMeta}>· {date}</span>
        {c.basis === "official" && c.verified === false ? <span className={s.chipMeta}>· not verified</span> : null}
      </button>
      {open ? (
        <div id={id} className={s.chipDetail} data-testid="ask-chip-detail">
          <p>{c.headline}</p>
          <p>
            {BASIS[c.basis]}
            {c.basis === "official" ? (c.verified ? " · verified by a person" : c.verified === false ? " · transcribed by Neritic, not yet verified" : "") : ""} · as of {c.as_of ? date : "unknown"}
          </p>
          <ul className={s.links}>
            <li>
              {c.source_url ? (
                <a href={c.source_url} target={/^https?:/.test(c.source_url) ? "_blank" : undefined} rel="noopener noreferrer" className={`${s.link} ${s.linkRow}`}>
                  Open {c.source_name}
                </a>
              ) : (
                <span className={s.linkRow}>Neritic&rsquo;s own analysis (no external page)</span>
              )}
            </li>
            <li>
              <a href={c.today_href} className={`${s.link} ${s.linkRow}`} data-testid="ask-see-today">
                See it on Today
              </a>
            </li>
          </ul>
        </div>
      ) : null}
    </li>
  );
}

function Answer({ a }: { a: AskAnswer }) {
  const refusal = a.outcome === "safety";
  return (
    <div className={s.answer} data-testid="ask-answer" data-outcome={a.outcome} data-mode={a.mode}>
      {a.preface.length ? (
        <div className={s.lines} data-testid="ask-preface">
          {a.preface.map((p, i) => (
            <p key={i} className={s.plain}>
              {p}
            </p>
          ))}
        </div>
      ) : null}
      {refusal && a.referrals.length ? (
        <ul className={s.links} aria-label="Official sources to check" data-testid="ask-referrals">
          {a.referrals.map((r) => (
            <li key={r.url}>
              <a href={r.url} target="_blank" rel="noopener noreferrer" className={`${s.link} ${s.linkRow}`}>
                {r.label}
              </a>{" "}
              <span className={s.by}>{r.why}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {a.lines.length ? (
        <div className={s.lines} data-ruled={refusal ? "" : undefined} data-testid="ask-lines">
          {refusal ? <p className={s.kicker}>For information only</p> : null}
          {a.lines.map((l, i) => {
            const nums = l.cites.map((id) => a.citations.findIndex((c) => c.fact_id === id) + 1).filter((n) => n > 0);
            // When the first line is the answer itself it is set as the answer, and the lines under it support it. An answer
            // that must open with the official status (a test or bloom question) keeps every line in the same weight: nothing
            // measured or forecast is ever set above an official record. The closing "not yet verified" line is a quiet note.
            const lead = i === 0 && !refusal && !!l.primary;
            const note = UNVERIFIED_LINE.test(l.text);
            const plain = !refusal && (l.primary || !a.lines.some((x) => x.primary) || !a.lines[0].primary);
            return (
              <p key={i} className={lead ? s.lead : note ? s.by : plain ? s.plain : undefined} data-testid={lead ? "ask-lead" : undefined}>
                {l.text}
                {l.source ? (
                  <span className={s.by} data-testid="ask-by">
                    {" "}
                    {l.source}
                  </span>
                ) : null}
                {nums.length ? (
                  <span className={s.cite} aria-label={`sources ${nums.join(", ")}`}>
                    [{nums.join(",")}]
                  </span>
                ) : null}
              </p>
            );
          })}
        </div>
      ) : null}
      {a.citations.length ? (
        <div>
          <p className={s.kicker}>Sources</p>
          <ul className={s.chips} data-testid="ask-citations">
            {a.citations.map((c, i) => (
              <Chip key={c.fact_id} c={c} i={i} />
            ))}
          </ul>
        </div>
      ) : null}
      {a.data_age ? (
        <div className={s.by} data-testid="ask-data-age">
          <p>
            <strong>Data age:</strong> {a.data_age.label}
          </p>
          {a.data_age.stale.map((x) => (
            <p key={x} className={s.old} data-testid="ask-stale">
              <VisuallyHidden>Out of date: </VisuallyHidden>
              {x}
            </p>
          ))}
        </div>
      ) : null}
      {a.caveats.length ? (
        <ul className={s.links}>
          {a.caveats.map((c) => (
            <li key={c} className={s.by}>
              {c}
            </li>
          ))}
        </ul>
      ) : null}
      {!refusal && a.referrals.length ? (
        <ul className={s.links}>
          {a.referrals.map((r) => (
            <li key={r.url}>
              <a href={r.url} target="_blank" rel="noopener noreferrer" className={`${s.link} ${s.linkRow}`}>
                {r.label}
              </a>
            </li>
          ))}
        </ul>
      ) : null}
      {a.notice ? (
        // the standing notice: after the answer, said once per conversation, quietly
        <p className={`${s.by} ${s.ruled}`} data-testid="ask-notice">
          {a.notice}
        </p>
      ) : null}
      <p className={s.by} data-testid="ask-made">
        {a.mode === "live" && a.model ? `Written by Claude (${a.model}) from the records above and checked by Neritic.` : a.mode === "live_fallback" ? "Assembled from Neritic's records without an AI model (live mode fallback)." : "Assembled from Neritic's records without an AI model."}
        {a.dataset === "fixture" ? ` ${SAMPLE_DATA_LABEL}.` : ""}
      </p>
    </div>
  );
}
