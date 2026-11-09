"use client";

import "@/pilot/today/today.css";
import "@/pilot/shell/pilot.css";
import Link from "next/link";
import { useEffect, useId, useRef, useState } from "react";
import { PILOT_PORTS } from "../../contracts";
import { readTodayPlace } from "../../today/navPlace";
import { READINESS_WORDS } from "../../readiness";
import { segments } from "../sms";
import { suggestionsFor } from "../suggestions";
import { DEMO_LABEL, DISCLAIMER, LIVE_LABEL, SAMPLE_DATA_LABEL } from "../templates";
import type { AskAnswer, AskMeta, Citation, SeenSnapshot } from "../types";

/**
 * Ask Neritic, the calm chat, in the pilot's Almanac voice (pilot.css). Used full-page on /ask (its
 * header is the page's hero band) and compact inside <AskSheet> on Today. Every answer shows its
 * sources as tappable chips (source link + the Today card), the age of its data, and how it was made
 * (assembled without a model, or written by Claude and checked).
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
    <p className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[13.5px] leading-snug text-n-ink-2" data-testid="ask-readiness">
      <span className="pilot-badge" data-readiness={meta.readiness.ask}>
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
      <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen((v) => !v)} className="ask-chip nx-focus nx-tint" data-testid="ask-chip">
        <i aria-hidden>{i + 1}</i>
        <span className="truncate">{c.source_name}</span>
        <span className="whitespace-nowrap text-n-ink-3">· {date}</span>
        {c.basis === "official" && c.verified === false ? <span className="whitespace-nowrap text-n-official">· not verified</span> : null}
      </button>
      {open ? (
        <div id={id} className="mt-2 border-l-2 border-n-line-strong py-0.5 pl-3 text-[14px] leading-relaxed text-n-ink-2" data-testid="ask-chip-detail">
          <p className="text-n-ink">{c.headline}</p>
          <p className="mt-1">
            {BASIS[c.basis]}
            {c.basis === "official" ? (c.verified ? " · verified by a person" : c.verified === false ? " · transcribed by Neritic, not yet verified" : "") : ""} · as of {c.as_of ?? "unknown"}
          </p>
          <p className="mt-1 flex flex-wrap gap-x-5">
            {c.source_url ? (
              <a href={c.source_url} target={/^https?:/.test(c.source_url) ? "_blank" : undefined} rel="noopener noreferrer" className="pilot-link nx-focus inline-flex min-h-11 items-center">
                Open {c.source_name} ↗
              </a>
            ) : (
              <span className="inline-flex min-h-11 items-center">Neritic&apos;s own analysis (no external page)</span>
            )}
            <a href={c.today_href} className="pilot-link nx-focus inline-flex min-h-11 items-center" data-testid="ask-see-today">
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
    <div className={`ask-a space-y-3 ${safety ? "safety" : ""}`} data-testid="ask-answer" data-outcome={a.outcome} data-mode={a.mode}>
      {a.notice ? (
        <p className="nx-margin" data-testid="ask-notice">
          {a.notice}
        </p>
      ) : null}
      {a.preface.length ? (
        <div className="space-y-1.5 text-[16px] leading-relaxed text-n-ink" data-testid="ask-preface">
          {a.preface.map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </div>
      ) : null}
      {safety && a.referrals.length ? (
        <ul aria-label="Official sources to check" data-testid="ask-referrals">
          {a.referrals.map((r) => (
            <li key={r.url} className="flex flex-wrap items-baseline gap-x-2">
              <a href={r.url} target="_blank" rel="noopener noreferrer" className="pilot-link nx-focus inline-flex min-h-11 items-center font-medium">
                {r.label} ↗
              </a>
              <span className="text-[13.5px] text-n-ink-3">{r.why}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {a.lines.length ? (
        <div className={`space-y-2.5 text-[16px] leading-relaxed ${safety ? "border-t border-n-line pt-3 text-n-ink-2" : "text-n-ink"}`} data-testid="ask-lines">
          {safety ? <p className="nx-caps">For information only</p> : null}
          {a.lines.map((l, i) => {
            const nums = l.cites.map((id) => a.citations.findIndex((c) => c.fact_id === id) + 1).filter((n) => n > 0);
            return (
              <p key={i}>
                {l.text}
                {nums.length ? (
                  <span className="ml-1 font-data text-[11px] text-n-ink-3" aria-label={`sources ${nums.join(", ")}`}>
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
          <p className="nx-caps mb-2">Sources</p>
          <ul className="flex flex-wrap gap-2" data-testid="ask-citations">
            {a.citations.map((c, i) => (
              <CitationChip key={c.fact_id} c={c} i={i} />
            ))}
          </ul>
        </div>
      ) : null}
      {a.data_age ? (
        <div className="today-src" data-testid="ask-data-age">
          <p>
            <span className="font-medium text-n-ink-2">Data age:</span> {a.data_age.label}
          </p>
          {a.data_age.stale.map((s) => (
            <p key={s} className="t-amber" data-testid="ask-stale">
              <span className="sr-only">Out of date: </span>
              {s}
            </p>
          ))}
        </div>
      ) : null}
      {a.caveats.length ? (
        <ul className="nx-margin space-y-1">
          {a.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      ) : null}
      {!safety && a.referrals.length ? (
        <p className="flex flex-wrap gap-x-5 text-[14px]">
          {a.referrals.map((r) => (
            <a key={r.url} href={r.url} target="_blank" rel="noopener noreferrer" className="pilot-link nx-focus inline-flex min-h-11 items-center">
              {r.label} ↗
            </a>
          ))}
        </p>
      ) : null}
      <p className="today-src" data-testid="ask-made">
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
    <aside aria-labelledby="ask-sms-title" className="today-sec self-start lg:sticky lg:top-4" data-testid="ask-sms-preview">
      <header>
        <h2 id="ask-sms-title" className="today-h2">
          SMS preview
        </h2>
      </header>
      <p className="mt-3 text-[14px] t-amber" data-testid="ask-sms-label">
        SMS in carrier review — preview only
      </p>
      {text ? (
        <>
          <p className="ask-sms mt-3" data-testid="ask-sms-text">
            {text}
          </p>
          <p className="today-src mt-2">
            {text.length} characters · {segments(text)} SMS segment{segments(text) === 1 ? "" : "s"}
          </p>
        </>
      ) : (
        <p className="today-src mt-3">Ask a question to see the exact reply a phone would get.</p>
      )}
    </aside>
  );
}

export type AskChatProps = {
  initialPlace: string;
  meta?: AskMeta;
  suggestions?: string[];
  /** inside the Today sheet: no hero band, no SMS column */
  compact?: boolean;
  /** heading id for the title (the sheet labels its dialog with it) */
  headingId?: string;
  /** no port was named in the link: open on the port the reader last opened in Today, if it is a pilot port */
  preferSaved?: boolean;
};

export function AskChat({ initialPlace, meta: metaProp, suggestions: sugProp, compact = false, headingId, preferSaved = false }: AskChatProps) {
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
    if (!preferSaved) return;
    const saved = readTodayPlace();
    if (saved && PILOT_PORTS.some((p) => p.place_id === saved)) setPlace(saved);
  }, [preferSaved]);

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
  const titleId = headingId ?? "ask-title";
  const Sub = compact ? "h3" : "h2";

  const controls = (
    <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2">
      <label htmlFor={`${inputId}-port`} className="nx-caps">
        Port
      </label>
      <select id={`${inputId}-port`} value={place} onChange={(e) => setPlace(e.target.value)} className="pilot-field nx-focus !w-auto !min-h-11" data-testid="ask-port">
        {PILOT_PORTS.map((p) => (
          <option key={p.place_id} value={p.place_id}>
            {p.label}
          </option>
        ))}
      </select>
      {meta?.dataset === "fixture" ? (
        <span className="pilot-badge sample" data-testid="ask-sample">
          Sample data
        </span>
      ) : null}
      {!compact && (
        <Link href={`/today?place=${place}`} className="today-btn nx-focus nx-press nx-tint" data-testid="ask-today-link">
          Today at {label}
        </Link>
      )}
    </div>
  );

  const header = compact ? (
    <header>
      <p className="nx-caps">Answers from today&rsquo;s facts · {label}</p>
      <h2 id={titleId} className="mt-1 font-serif text-[28px] leading-tight text-n-ink">
        Ask Neritic
      </h2>
      <div className="mt-2">{meta ? <ReadinessBadge meta={meta} /> : null}</div>
      {controls}
    </header>
  ) : (
    <header className="today-hero" data-testid="pilot-hero">
      <div className="today-wrap relative pb-6 pt-5 md:pb-9 md:pt-8">
        <p className="nx-caps">Neritic private pilot · answers from today&rsquo;s facts</p>
        <h1 id={titleId} className="pilot-title mt-2 text-n-ink">
          Ask Neritic
        </h1>
        <p className="pilot-lede mt-3">A plain question about {label}, answered only from the records on its Today page. Every line names its source and its date.</p>
        <div className="mt-4">{meta ? <ReadinessBadge meta={meta} /> : null}</div>
        {controls}
      </div>
    </header>
  );

  const chat = (
    <section aria-labelledby={titleId} className="min-w-0">
      {turns.length === 0 ? (
        <div className={compact ? "mt-4" : "today-sec"}>
          <p className="pilot-note" data-testid="ask-disclaimer">
            {DISCLAIMER}
          </p>
          {meta?.logging ? <p className="today-src mt-2">Questions are logged for 90 days to check answers, then deleted.</p> : null}
        </div>
      ) : null}

      <ol className={`space-y-8 ${turns.length ? (compact ? "mt-5" : "pt-7") : ""}`} aria-live="polite" aria-busy={busy} data-testid="ask-turns">
        {turns.map((t, i) => (
          <li key={i} className="space-y-3">
            <div>
              <p className="nx-caps" aria-hidden>
                You asked
              </p>
              <p className="ask-q mt-1" data-testid="ask-question">
                <span className="sr-only">You asked: </span>
                {t.q}
              </p>
            </div>
            {t.a ? (
              <AnswerCard a={t.a} />
            ) : t.error ? (
              <p role="alert" className="pilot-note red">
                {t.error}
              </p>
            ) : (
              <p className="today-src">Checking Neritic&apos;s records for {label}…</p>
            )}
          </li>
        ))}
      </ol>
      <div ref={endRef} />

      <div className={compact ? "mt-6" : "today-sec"}>
        <header className={compact ? "border-b border-n-line pb-2" : undefined}>
          <Sub className="today-h2" id={`${inputId}-sug`}>
            Common questions for {label}
          </Sub>
        </header>
        <ul className="mt-3 flex flex-wrap gap-2" aria-labelledby={`${inputId}-sug`} data-testid="ask-suggestions">
          {suggestions.map((s) => (
            <li key={s}>
              <button type="button" disabled={busy} onClick={() => ask(s)} className="ask-suggest nx-focus nx-tint disabled:opacity-60">
                {s}
              </button>
            </li>
          ))}
        </ul>
      </div>

      <form
        className="mt-5 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void ask(q);
        }}
      >
        <label htmlFor={inputId} className="sr-only">
          Ask about {label}
        </label>
        <input id={inputId} value={q} onChange={(e) => setQ(e.target.value)} maxLength={400} placeholder={`Ask about ${label}…`} autoComplete="off" className="pilot-field nx-focus min-w-0 flex-1 placeholder:text-n-ink-3" data-testid="ask-input" />
        <button type="submit" disabled={busy || !q.trim()} className="today-btn primary nx-focus nx-press !min-h-12 px-5 disabled:opacity-50" data-testid="ask-send">
          Ask
        </button>
      </form>
    </section>
  );

  if (compact)
    return (
      <div data-testid="ask-chat">
        {header}
        {chat}
      </div>
    );
  return (
    <div data-testid="ask-chat">
      {header}
      <div className="today-wrap">
        <div className="grid gap-x-12 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
          {chat}
          {meta ? <SmsPreview text={last?.sms ?? null} meta={meta} /> : null}
        </div>
      </div>
    </div>
  );
}
