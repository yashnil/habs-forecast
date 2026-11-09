"use client";

import { useEffect, useId, useRef, useState } from "react";
import { PILOT_PORTS } from "../../contracts";
import { READINESS_WORDS } from "../../readiness";
import { segments } from "../sms";
import { suggestionsFor } from "../suggestions";
import { DEMO_LABEL, DISCLAIMER, LIVE_LABEL, SAMPLE_DATA_LABEL } from "../templates";
import type { AskAnswer, AskMeta, Citation, SeenSnapshot } from "../types";

/**
 * Ask Neritic, the calm chat. Used full-page on /ask and compact inside <AskSheet>.
 * Every answer shows its sources as tappable chips (source link + the Today card), the age of its
 * data, and how it was made (assembled without a model, or written by Claude and checked).
 */

type Turn = { q: string; a: AskAnswer | null; error?: string };

const SEEN_KEY = (place: string) => `nx-ask-seen-v1:${place}`;
function readSeen(place: string): SeenSnapshot | null {
  try {
    const raw = localStorage.getItem(SEEN_KEY(place));
    return raw ? (JSON.parse(raw) as SeenSnapshot) : null;
  } catch {
    return null;
  }
}
function writeSeen(place: string, s: SeenSnapshot) {
  try {
    localStorage.setItem(SEEN_KEY(place), JSON.stringify(s));
  } catch {
    /* storage unavailable: "what changed" just compares nothing */
  }
}

const BASIS: Record<Citation["basis"], string> = {
  official: "Official record",
  measured: "Measurement",
  forecast: "Forecast",
  outlook: "Outlook",
  analysis: "Analysis",
  derived: "Neritic",
};

function ReadinessBadge({ meta }: { meta: AskMeta }) {
  const live = meta.readiness.ask === "live";
  return (
    <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] leading-snug text-n-ink-2" data-testid="ask-readiness">
      <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[12px] font-medium ${live ? "border-n-measured-line text-n-measured" : "border-n-line-strong text-n-ink"}`}>
        {READINESS_WORDS[meta.readiness.ask]}
      </span>
      <span>{live ? LIVE_LABEL : DEMO_LABEL}</span>
    </p>
  );
}

function CitationChip({ c, i }: { c: Citation; i: number }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const date = c.as_of?.slice(0, 10) ?? "no date";
  return (
    <li className="min-w-0">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex min-h-[36px] max-w-full items-center gap-1.5 rounded-full border border-n-line-strong bg-n-surface-2 px-3 py-1 text-left text-[13px] text-n-ink hover:border-n-accent focus-visible:outline-2 focus-visible:outline-n-accent"
        data-testid="ask-chip"
      >
        <span aria-hidden className="font-[family-name:var(--nx-font-data)] text-[11px] text-n-ink-3">
          {i + 1}
        </span>
        <span className="truncate">{c.source_name}</span>
        <span className="text-n-ink-3">· {date}</span>
        {c.basis === "official" && c.verified === false ? <span className="text-n-official">· not verified</span> : null}
      </button>
      {open ? (
        <div id={id} className="mt-2 rounded-lg border border-n-line bg-n-surface-2 p-3 text-[13px] leading-relaxed text-n-ink-2" data-testid="ask-chip-detail">
          <p className="text-n-ink">{c.headline}</p>
          <p className="mt-1">
            {BASIS[c.basis]}
            {c.basis === "official" ? (c.verified ? " · verified by a person" : " · transcribed by Neritic, not yet verified") : ""} · as of {c.as_of ?? "unknown"}
          </p>
          <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
            {c.source_url ? (
              <a href={c.source_url} target="_blank" rel="noopener noreferrer" className="text-n-accent underline underline-offset-2">
                Open {c.source_name} ↗
              </a>
            ) : (
              <span>Neritic&apos;s own analysis (no external page)</span>
            )}
            <a href={c.today_href} className="text-n-accent underline underline-offset-2">
              See it on Today
            </a>
          </p>
        </div>
      ) : null}
    </li>
  );
}

function AnswerCard({ a }: { a: AskAnswer }) {
  const safety = a.outcome === "safety";
  return (
    <div className={`space-y-3 rounded-xl border p-4 ${safety ? "border-n-official-line bg-n-official-bg" : "border-n-line bg-n-surface"}`} data-testid="ask-answer" data-outcome={a.outcome} data-mode={a.mode}>
      {a.notice ? (
        <p className="rounded-lg border border-n-line-strong p-3 text-[13px] leading-relaxed text-n-ink-2" data-testid="ask-notice">
          {a.notice}
        </p>
      ) : null}
      {a.preface.length ? (
        <div className="space-y-1.5 text-[15px] leading-relaxed text-n-ink" data-testid="ask-preface">
          {a.preface.map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </div>
      ) : null}
      {safety && a.referrals.length ? (
        <ul className="space-y-1.5" aria-label="Official sources to check" data-testid="ask-referrals">
          {a.referrals.map((r) => (
            <li key={r.url}>
              <a href={r.url} target="_blank" rel="noopener noreferrer" className="font-medium text-n-accent underline underline-offset-2">
                {r.label} ↗
              </a>
              <span className="text-[13px] text-n-ink-3"> — {r.why}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {a.lines.length ? (
        <div className={`space-y-2 text-[15px] leading-relaxed ${safety ? "border-t border-n-line pt-3 text-n-ink-2" : "text-n-ink"}`} data-testid="ask-lines">
          {safety ? <p className="text-[12px] font-medium uppercase tracking-wide text-n-ink-3">For information only</p> : null}
          {a.lines.map((l, i) => {
            const nums = l.cites.map((id) => a.citations.findIndex((c) => c.fact_id === id) + 1).filter((n) => n > 0);
            return (
              <p key={i}>
                {l.text}
                {nums.length ? (
                  <span className="ml-1 font-[family-name:var(--nx-font-data)] text-[11px] text-n-ink-3" aria-label={`sources ${nums.join(", ")}`}>
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
          <p className="mb-1.5 text-[12px] font-medium text-n-ink-3">Sources</p>
          <ul className="flex flex-wrap gap-2" data-testid="ask-citations">
            {a.citations.map((c, i) => (
              <CitationChip key={c.fact_id} c={c} i={i} />
            ))}
          </ul>
        </div>
      ) : null}
      {a.data_age ? (
        <div className="text-[13px] leading-relaxed text-n-ink-2" data-testid="ask-data-age">
          <p>
            <span className="font-medium text-n-ink">Data age:</span> {a.data_age.label}
          </p>
          {a.data_age.stale.map((s) => (
            <p key={s} className="text-n-stale" data-testid="ask-stale">
              <span aria-hidden>⚠ </span>
              <span className="sr-only">Out of date: </span>
              {s}
            </p>
          ))}
        </div>
      ) : null}
      {a.caveats.length ? (
        <ul className="space-y-1 text-[13px] leading-relaxed text-n-ink-3">
          {a.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      ) : null}
      {!safety && a.referrals.length ? (
        <p className="flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
          {a.referrals.map((r) => (
            <a key={r.url} href={r.url} target="_blank" rel="noopener noreferrer" className="text-n-accent underline underline-offset-2">
              {r.label} ↗
            </a>
          ))}
        </p>
      ) : null}
      <p className="text-[12px] text-n-ink-3" data-testid="ask-made">
        {a.mode === "live" && a.model
          ? `Written by Claude (${a.model}) from the records above and checked by Neritic.`
          : a.mode === "live_fallback"
            ? "Assembled from Neritic's records without an AI model (live mode fallback)."
            : "Assembled from Neritic's records without an AI model."}
        {a.dataset === "fixture" ? ` ${SAMPLE_DATA_LABEL}.` : ""}
      </p>
    </div>
  );
}

function SmsPreview({ text, meta }: { text: string | null; meta: AskMeta }) {
  if (meta.readiness.sms === "live") return null;
  return (
    <aside aria-labelledby="ask-sms-title" className="self-start rounded-xl border border-n-line bg-n-surface p-4 lg:sticky lg:top-6" data-testid="ask-sms-preview">
      <h2 id="ask-sms-title" className="text-[14px] font-medium text-n-ink">
        SMS preview
      </h2>
      <p className="mt-0.5 text-[13px] text-n-stale" data-testid="ask-sms-label">
        SMS in carrier review — preview only
      </p>
      {text ? (
        <>
          <p className="mt-3 whitespace-pre-wrap rounded-2xl rounded-bl-sm bg-n-surface-3 p-3 text-[14px] leading-snug text-n-ink" data-testid="ask-sms-text">
            {text}
          </p>
          <p className="mt-1.5 text-[12px] text-n-ink-3">
            {text.length} characters · {segments(text)} SMS segment{segments(text) === 1 ? "" : "s"}
          </p>
        </>
      ) : (
        <p className="mt-3 text-[13px] text-n-ink-3">Ask a question to see the exact reply a phone would get.</p>
      )}
    </aside>
  );
}

export type AskChatProps = {
  initialPlace: string;
  meta?: AskMeta;
  suggestions?: string[];
  compact?: boolean;
  /** heading level for the title (the sheet uses its own) */
  headingId?: string;
};

export function AskChat({ initialPlace, meta: metaProp, suggestions: sugProp, compact = false, headingId }: AskChatProps) {
  const [place, setPlace] = useState(initialPlace);
  const [meta, setMeta] = useState<AskMeta | null>(metaProp ?? null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const prevSeen = useRef<SeenSnapshot | null>(null);
  const inputId = useId();
  const endRef = useRef<HTMLDivElement>(null);
  const label = PILOT_PORTS.find((p) => p.place_id === place)?.label ?? "your port";

  const suggestions = sugProp && place === initialPlace ? sugProp : suggestionsFor(place);

  useEffect(() => {
    prevSeen.current = readSeen(place);
  }, [place]);

  useEffect(() => {
    if (metaProp) return;
    let off = false;
    fetch(`/api/ask?place=${encodeURIComponent(initialPlace)}`)
      .then((r) => r.json())
      .then((m: AskMeta) => {
        if (!off) setMeta({ readiness: m.readiness, logging: m.logging, dataset: m.dataset });
      })
      .catch(() => {});
    return () => {
      off = true;
    };
  }, [initialPlace, metaProp]);

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy) return;
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
  const Title = compact ? "h2" : "h1";

  return (
    <div className={`grid gap-6 ${compact ? "" : "lg:grid-cols-[minmax(0,1fr)_320px]"}`} data-testid="ask-chat">
      <section aria-labelledby={headingId ?? "ask-title"} className="min-w-0 space-y-4">
        <header className="space-y-2">
          <Title id={headingId ?? "ask-title"} className={`font-[family-name:var(--nx-font-display)] font-medium text-n-ink ${compact ? "text-[24px]" : "text-[32px] leading-tight sm:text-[40px]"}`}>
            Ask Neritic
          </Title>
          {meta ? <ReadinessBadge meta={meta} /> : null}
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor={`${inputId}-port`} className="text-[14px] text-n-ink-2">
              Port
            </label>
            <select
              id={`${inputId}-port`}
              value={place}
              onChange={(e) => setPlace(e.target.value)}
              className="min-h-[40px] rounded-lg border border-n-line-strong bg-n-surface-2 px-2 text-[15px] text-n-ink"
              data-testid="ask-port"
            >
              {PILOT_PORTS.map((p) => (
                <option key={p.place_id} value={p.place_id}>
                  {p.label}
                </option>
              ))}
            </select>
            {meta?.dataset === "fixture" ? (
              <span className="rounded-full border border-n-stale/50 px-2 py-0.5 text-[12px] text-n-stale" data-testid="ask-sample">
                Sample data
              </span>
            ) : null}
          </div>
        </header>

        {turns.length === 0 ? (
          <div className="space-y-3 rounded-xl border border-n-line bg-n-surface p-4">
            <p className="text-[14px] leading-relaxed text-n-ink-2" data-testid="ask-disclaimer">
              {DISCLAIMER}
            </p>
            {meta?.logging ? <p className="text-[13px] text-n-ink-3">Questions are logged for 90 days to check answers, then deleted.</p> : null}
          </div>
        ) : null}

        <ol className="space-y-5" aria-live="polite" aria-busy={busy} data-testid="ask-turns">
          {turns.map((t, i) => (
            <li key={i} className="space-y-2">
              <p className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-sm bg-n-accent-bg px-3.5 py-2 text-[15px] text-n-ink" data-testid="ask-question">
                <span className="sr-only">You asked: </span>
                {t.q}
              </p>
              {t.a ? (
                <AnswerCard a={t.a} />
              ) : t.error ? (
                <p role="alert" className="rounded-xl border border-n-down/50 p-3 text-[14px] text-n-down">
                  {t.error}
                </p>
              ) : (
                <p className="text-[14px] text-n-ink-3">Checking Neritic&apos;s records for {label}…</p>
              )}
            </li>
          ))}
        </ol>
        <div ref={endRef} />

        <div className="space-y-2">
          <p className="text-[13px] text-n-ink-3" id={`${inputId}-sug`}>
            Common questions for {label}
          </p>
          <ul className="flex flex-wrap gap-2" aria-labelledby={`${inputId}-sug`} data-testid="ask-suggestions">
            {suggestions.map((s) => (
              <li key={s}>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => ask(s)}
                  className="min-h-[40px] rounded-full border border-n-line-strong px-3 py-1.5 text-left text-[14px] text-n-ink-2 hover:border-n-accent hover:text-n-ink disabled:opacity-60"
                >
                  {s}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void ask(q);
          }}
        >
          <label htmlFor={inputId} className="sr-only">
            Ask about {label}
          </label>
          <input
            id={inputId}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            maxLength={400}
            placeholder={`Ask about ${label}…`}
            autoComplete="off"
            className="min-h-[44px] min-w-0 flex-1 rounded-lg border border-n-line-strong bg-n-surface-2 px-3 text-[16px] text-n-ink placeholder:text-n-ink-3"
            data-testid="ask-input"
          />
          <button type="submit" disabled={busy || !q.trim()} className="min-h-[44px] rounded-lg bg-n-accent px-4 text-[15px] font-medium text-n-on-accent disabled:opacity-60" data-testid="ask-send">
            Ask
          </button>
        </form>
      </section>
      {meta ? <SmsPreview text={last?.sms ?? null} meta={meta} /> : null}
    </div>
  );
}
