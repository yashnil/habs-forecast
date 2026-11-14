"use client";

import { useId, useRef, type ReactNode } from "react";

/**
 * A small explanation panel (native popover: Escape and an outside tap close it, and its words stay in the
 * DOM). Hung under its trigger on a wide screen; a bottom sheet on a phone. Used by Unverified and
 * Freshness; pages may use it for an explanation of their own.
 */
export function Pop({
  trigger,
  label,
  children,
  triggerClass = "",
  testId,
}: {
  /** what the trigger shows */
  trigger: ReactNode;
  /** the panel's accessible name */
  label: string;
  children: ReactNode;
  triggerClass?: string;
  testId?: string;
}) {
  const id = useId();
  const btn = useRef<HTMLButtonElement | null>(null);
  return (
    <>
      <button ref={btn} type="button" popoverTarget={id} aria-haspopup="dialog" className={triggerClass} data-testid={testId}>
        {trigger}
      </button>
      <div
        id={id}
        popover="auto"
        role="dialog"
        aria-label={label}
        className="nx-pop-panel"
        data-testid={testId ? `${testId}-pop` : undefined}
        onToggle={(e) => {
          const box = e.currentTarget;
          if (e.newState !== "open" || !btn.current) return;
          // a phone: the CSS sheet at the bottom edge
          if (matchMedia("(max-width: 639px)").matches) return;
          const t = btn.current.getBoundingClientRect();
          const p = box.getBoundingClientRect();
          const left = Math.min(Math.max(12, t.left), window.innerWidth - p.width - 12);
          const below = t.bottom + 8 + p.height <= window.innerHeight - 12;
          box.style.left = `${left}px`;
          box.style.top = `${below ? t.bottom + 8 : Math.max(12, t.top - 8 - p.height)}px`;
        }}
      >
        {children}
      </div>
    </>
  );
}
