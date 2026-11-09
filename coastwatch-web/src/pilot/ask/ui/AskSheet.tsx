"use client";

import { useEffect, useRef } from "react";
import { AskChat } from "./AskChat";

/**
 * A compact Ask Neritic in a modal sheet, for the Today page's "Ask Neritic" action:
 *
 *   const [ask, setAsk] = useState(false);
 *   <button onClick={() => setAsk(true)}>Ask Neritic</button>
 *   <AskSheet place="port-592" open={ask} onClose={() => setAsk(false)} />
 *
 * Native <dialog>: focus is trapped and Escape closes it. Readiness and suggestions are fetched
 * from GET /api/ask when it opens.
 */
export function AskSheet({ place, open, onClose }: { place: string; open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      aria-labelledby="ask-sheet-title"
      className="m-0 mt-auto max-h-[92dvh] w-full max-w-none overflow-y-auto rounded-t-2xl border border-n-line bg-n-bg p-4 font-[family-name:var(--nx-font-ui)] text-n-ink backdrop:bg-black/60 sm:m-auto sm:max-w-[640px] sm:rounded-2xl sm:p-6"
      data-testid="ask-sheet"
    >
      <div className="mb-2 flex justify-end">
        <button type="button" onClick={onClose} className="min-h-[40px] rounded-lg border border-n-line-strong px-3 text-[14px] text-n-ink-2" aria-label="Close Ask Neritic">
          Close
        </button>
      </div>
      {open ? <AskChat initialPlace={place} compact headingId="ask-sheet-title" /> : null}
    </dialog>
  );
}
