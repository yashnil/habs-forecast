"use client";

import { useEffect, useId, useState } from "react";
import { usePilotAccount } from "./context";
import { feedbackFor, saveFeedback, type FeedbackRecord } from "./records";
import { sendFeedback } from "./sync";

/**
 * "Was this useful?" for one PortFact (by its stable id). Kept on this device; also sent to the
 * signed-in account's feedback rows. Mount under any fact card:
 *     <Feedback factId={fact.id} placeId={fact.place_id} />
 */
export function Feedback({ factId, placeId = null }: { factId: string; placeId?: string | null }) {
  const { session } = usePilotAccount();
  const [given, setGiven] = useState<FeedbackRecord | null>(null);
  const [note, setNote] = useState("");
  const [noteOpen, setNoteOpen] = useState(false);
  const [sent, setSent] = useState<"device" | "account" | null>(null);
  const noteId = useId();
  useEffect(() => setGiven(feedbackFor(factId)), [factId]);

  async function answer(useful: boolean, withNote: string | null = null) {
    const r: FeedbackRecord = { fact_id: factId, place_id: placeId, useful, note: withNote?.trim() ? withNote.trim().slice(0, 1000) : null, at: new Date().toISOString() };
    saveFeedback(r);
    setGiven(r);
    setSent("device");
    if (session.client && session.status === "signed_in") {
      const res = await sendFeedback(session.client, r);
      if (res.ok) setSent("account");
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[13.5px] text-n-ink-2" data-testid="feedback" data-fact={factId}>
      <span id={`${noteId}-q`}>Was this useful?</span>
      <div role="group" aria-labelledby={`${noteId}-q`} className="flex gap-1">
        {[
          { v: true, label: "Yes" },
          { v: false, label: "No" },
        ].map((o) => (
          <button
            key={o.label}
            type="button"
            aria-pressed={given?.useful === o.v}
            onClick={() => {
              void answer(o.v, note || null);
              if (!o.v) setNoteOpen(true);
            }}
            className={`nx-focus min-h-11 min-w-11 rounded-full border px-3 font-medium ${given?.useful === o.v ? "border-n-accent bg-n-accent-bg text-n-ink" : "border-n-line-strong text-n-ink-2"}`}
            data-testid={`feedback-${o.label.toLowerCase()}`}
          >
            {o.label}
          </button>
        ))}
      </div>
      {given && !noteOpen && (
        <button type="button" onClick={() => setNoteOpen(true)} className="nx-focus min-h-11 text-n-accent underline underline-offset-2">
          Add a note
        </button>
      )}
      {noteOpen && given && (
        <form
          className="flex w-full flex-wrap items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void answer(given.useful, note);
            setNoteOpen(false);
          }}
        >
          <label htmlFor={noteId} className="w-full text-[13px]">
            What would make it more useful? (optional, no personal details)
          </label>
          <input id={noteId} value={note} maxLength={1000} onChange={(e) => setNote(e.target.value)} className="nx-focus min-h-11 min-w-0 flex-1 rounded-lg border border-n-line-strong bg-n-surface px-3 text-[15px] text-n-ink" data-testid="feedback-note" />
          <button type="submit" className="nx-press nx-focus min-h-11 rounded-lg bg-n-accent px-4 font-semibold text-n-on-accent">
            Send
          </button>
        </form>
      )}
      <span role="status" className="w-full text-[12.5px]">
        {sent === "account" ? "Thanks — saved to your account." : sent === "device" ? "Thanks — saved on this device." : ""}
      </span>
    </div>
  );
}
