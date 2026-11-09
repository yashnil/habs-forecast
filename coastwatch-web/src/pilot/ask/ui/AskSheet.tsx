"use client";

import "@/pilot/today/today.css";
import { useEffect, useRef } from "react";
import { Icon } from "@/v4/ui/Icon";
import { AskChat } from "./AskChat";

/**
 * Ask Neritic in place, for the Today page's "Ask Neritic" action: the same sheet as a status row's
 * detail (a bottom sheet on phones, a side panel on desktop; `today-sheet`).
 *
 *   const [ask, setAsk] = useState(false);
 *   <AskSheet place="port-592" open={ask} onClose={() => setAsk(false)} />
 *
 * Native <dialog>: focus is trapped and Escape closes it. A citation's "See it on Today" closes the
 * sheet first, so the reader lands on the card itself. Readiness and suggestions are fetched from
 * GET /api/ask when it opens.
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
      onClick={(e) => {
        // a tap on the backdrop closes; so does a link to a Today card (the card is under this sheet)
        const link = (e.target as HTMLElement).closest?.("a[href*='#fact-']");
        if (e.target === ref.current || link) onClose();
      }}
      aria-labelledby="ask-sheet-title"
      className="today-sheet"
      data-testid="ask-sheet"
    >
      {open ? (
        <div className="flex max-h-[88dvh] flex-col md:max-h-dvh">
          <div className="flex justify-end border-b border-n-line px-4 py-2">
            <button type="button" onClick={onClose} className="nx-focus nx-press grid h-11 w-11 place-items-center rounded-[3px] border border-n-line" aria-label="Close Ask Neritic" data-testid="ask-sheet-close">
              <Icon name="close" />
            </button>
          </div>
          <div className="overflow-y-auto px-4 pb-8 pt-4">
            <AskChat initialPlace={place} compact headingId="ask-sheet-title" />
          </div>
        </div>
      ) : null}
    </dialog>
  );
}
