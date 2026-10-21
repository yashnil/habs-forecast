"use client";

import { useId, useState, type ReactNode } from "react";
import { Icon } from "@/v4/ui/Icon";

/**
 * The v7 component language (docs/coastwatch/v7/design.md): a few deliberate primitives instead of
 * cards. Sounding (a value read off the instrument), Statement (one claim with its evidence family),
 * Note (a folded row: title, count, one-line summary; its body stays mounted so its wording is in the
 * page), Stepper (a number with big ±), Segmented (one choice of a few), ContourRule (the page's one
 * isobath hairline).
 */

export type Family = "official" | "model" | "measured" | "conditions" | "unknown";

export function Sounding({ label, value, unit, note, testId, tone }: { label: ReactNode; value: ReactNode; unit?: ReactNode; note?: ReactNode; testId?: string; tone?: "muted" }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5" data-testid={testId}>
      <span className="text-[13px] font-semibold leading-4 text-n-ink-2">{label}</span>
      <span className={`nx-sounding-v ${tone === "muted" ? "!text-n-ink-3" : ""}`}>
        {value}
        {unit && <span className="ml-1 text-[15px] font-semibold text-n-ink-2">{unit}</span>}
      </span>
      {note && <span className="nx-note-t">{note}</span>}
    </div>
  );
}

export function Statement({ family, kind, children, more, testId }: { family: Family; kind?: ReactNode; children: ReactNode; more?: ReactNode; testId?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className="flex gap-3 py-2.5" data-testid={testId} data-family={family}>
      <span aria-hidden className="nx-tick" data-family={family} />
      <div className="min-w-0 flex-1">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 text-[16px] leading-6 text-n-ink">{children}</div>
          {more && (
            <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)} className="nx-focus nx-tint -my-2 grid h-11 w-11 shrink-0 place-items-center rounded-full text-n-ink-3 hover:text-n-ink">
              <Icon name="chevronDown" className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`} />
              <span className="sr-only">{open ? "Less" : "Evidence"}</span>
            </button>
          )}
        </div>
        {kind && <p className="nx-note-t mt-0.5">{kind}</p>}
        {more && (
          <div id={id} hidden={!open} className="mt-2 text-[14.5px] leading-relaxed text-n-ink-2">
            {more}
          </div>
        )}
      </div>
    </div>
  );
}

/** A folded row. `open` for one that starts open; the body is always rendered (hidden by <details>). */
export function Note({ title, count, summary, children, testId, id, open = false }: { title: ReactNode; count?: number | null; summary?: ReactNode; children: ReactNode; testId?: string; id?: string; open?: boolean }) {
  return (
    <details className="nx-note group border-t border-n-line" data-testid={testId} id={id} open={open || undefined}>
      <summary className="nx-focus flex min-h-[56px] cursor-pointer items-center gap-3 py-3">
        <span className="min-w-0 flex-1">
          <span className="text-[16px] font-semibold text-n-ink">
            {title}
            {count != null && <span className="nx-num ml-2 font-medium text-n-ink-3">{count}</span>}
          </span>
          {summary && <span className="mt-0.5 block truncate text-[13.5px] text-n-ink-2 group-open:hidden">{summary}</span>}
        </span>
        <Icon name="chevronDown" className="nx-note-chev h-[18px] w-[18px] shrink-0 text-n-ink-3" />
      </summary>
      <div className="nx-note-body pb-5">{children}</div>
    </details>
  );
}

export function Segmented<T extends string>({ label, value, options, onChange, testId }: { label: string; value: T; options: { id: T; label: string }[]; onChange: (v: T) => void; testId?: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-xl bg-n-surface-2 p-1" data-testid={testId}>
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          role="radio"
          aria-checked={o.id === value}
          onClick={() => onChange(o.id)}
          className={`nx-focus min-h-10 rounded-lg px-3 text-[14px] font-semibold ${o.id === value ? "bg-n-surface text-n-ink shadow-[0_1px_2px_rgba(11,24,32,0.12)]" : "text-n-ink-2 hover:text-n-ink"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** The page's single decorative line: an isobath drawn across a section edge. */
export function ContourRule({ className = "" }: { className?: string }) {
  return (
    <svg aria-hidden className={`nx-contour-rule ${className}`} viewBox="0 0 1200 10" preserveAspectRatio="none">
      <path d="M0 6 C 120 2, 220 9, 360 5 S 600 1, 760 6 S 1000 9, 1200 4" fill="none" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
