"use client";

import { useId, useState, type ReactNode } from "react";
import { parseField, type LimitKind } from "@/lib/tripEconomics";

/**
 * My Coast's own small instrument pieces (v10, Ocean Instrument): a caps label with its family colour,
 * a reading in light mono with its unit small beside it, a numeric input set as a ruled line rather than
 * a box, a choice group, and a reference row that reads differently from anything the user typed.
 * Composed from the system vocabulary (.nx-caps, .nx-value, .nx-margin, .nx-rule); no new tokens.
 */

export type Family = "official" | "model" | "measured" | "conditions" | "ink";

const FAMILY_TEXT: Record<Family, string> = {
  official: "text-n-official",
  model: "text-n-model",
  measured: "text-n-measured",
  conditions: "text-n-conditions",
  ink: "text-n-ink-3",
};

/** A small-capital label; its family colour says which kind of value follows. */
export function Caps({ family = "ink", children, className = "", as: As = "span", id }: { family?: Family; children: ReactNode; className?: string; as?: "span" | "p" | "h2" | "h3" | "h4" | "div"; id?: string }) {
  return (
    <As id={id} className={`nx-caps leading-[1.35] ${FAMILY_TEXT[family]} ${className}`}>
      {children}
    </As>
  );
}

/** A reading: light mono figures, the unit small beside them. Never animated. */
export function Reading({ value, unit, size = "md", muted = false, testId, className = "" }: { value: ReactNode; unit?: ReactNode; size?: "sm" | "md" | "lg"; muted?: boolean; testId?: string; className?: string }) {
  // a size in className replaces the preset (two font sizes in one class list resolve by stylesheet order, not by intent)
  const s = /(^|\s)text-\[\d/.test(className) ? "" : size === "lg" ? "text-[40px] sm:text-[44px]" : size === "sm" ? "text-[22px]" : "text-[30px]";
  return (
    <span className={`nx-value whitespace-nowrap ${s} ${muted ? "text-n-ink-3" : ""} ${className}`} data-testid={testId}>
      {value}
      {unit != null && <small>{unit}</small>}
    </span>
  );
}

/** The DOM id of a worksheet field, so a "Needs …" link can bring it into view and focus it. */
export const fieldDomId = (key: string) => `econ-in-${key.replace(/[^A-Za-z0-9]+/g, "-")}`;

/**
 * A number typed by the user: one ruled line (dashed while a needed value is blank), mono figures, the
 * unit beside it. `value` is canonical; `scale` converts it for display (lb shown as kg). Every keystroke
 * reaches `onChange`; bad text is refused where it is typed and never becomes a number.
 */
export function NumIn({
  id,
  label,
  value,
  onChange,
  kind,
  unit,
  scale = 1,
  decimals = 2,
  needed = false,
  note,
  testId,
  wide = false,
  labelHidden = false,
}: {
  id: string;
  label: ReactNode;
  value: number | null;
  onChange: (v: number | null) => void;
  kind: LimitKind;
  unit?: string;
  scale?: number;
  decimals?: number;
  /** blank and waited for by a result: a dashed rule and "needed" beside the label */
  needed?: boolean;
  note?: ReactNode;
  testId?: string;
  wide?: boolean;
  labelHidden?: boolean;
}) {
  const uid = useId();
  const shown = value === null ? "" : String(Math.round(value * scale * 10 ** decimals) / 10 ** decimals);
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clamp, setClamp] = useState<string | null>(null);
  const msgId = `${uid}-msg`;
  const noteId = `${uid}-note`;
  const blank = value === null && (text ?? "") === "";
  return (
    <div className="flex min-w-0 flex-col gap-1">
      {/* label on the left, the value on its ruled line at the right (a worksheet row) */}
      <div className={labelHidden ? "" : "grid grid-cols-[minmax(0,1fr)_auto] items-end gap-x-3"}>
        <label htmlFor={id} className={labelHidden ? "sr-only" : "pb-2 text-[14px] leading-snug text-n-ink-2"}>
          {label}
          {needed && blank && !labelHidden && <span className="nx-caps ml-2 text-[10px] text-n-ink-2">needed</span>}
        </label>
        {/* the focus ring is the field's own: drawn around the ruled line and its unit, not offset from the bare input */}
        <div className={`flex items-baseline gap-1.5 rounded-[2px] border-b outline-offset-2 has-[input:focus-visible]:outline has-[input:focus-visible]:outline-2 has-[input:focus-visible]:outline-[var(--nx-accent)] ${wide ? "w-full" : "w-[8.5rem]"} ${error ? "border-n-down" : needed && blank ? "border-dashed border-n-ink-2" : "border-n-line-strong"} focus-within:border-solid focus-within:border-n-accent`}>
          <input
            id={id}
            type="text"
            inputMode="decimal"
            autoComplete="off"
            enterKeyHint="next"
            value={text ?? shown}
            aria-invalid={error ? true : undefined}
            aria-describedby={[error || clamp ? msgId : null, note ? noteId : null].filter(Boolean).join(" ") || undefined}
            data-testid={testId}
            onFocus={() => setText(shown)}
            onBlur={() => setText(null)}
            onChange={(e) => {
              const t = e.target.value;
              setText(t);
              const p = parseField(t, kind);
              setError(p.error);
              setClamp(null);
              if (p.error) return;
              if (p.value === null) onChange(null);
              else {
                onChange(p.value / scale);
                if (p.clamped) setClamp(`Limited to ${p.value.toLocaleString("en-US")}`);
              }
            }}
            className="min-h-11 w-full min-w-0 bg-transparent py-1 pr-0.5 text-right font-data text-[20px] font-light tabular-nums text-n-ink outline-none placeholder:text-n-ink-3"
            placeholder="—"
          />
          {unit && <span className="shrink-0 text-[12.5px] text-n-ink-2">{unit}</span>}
        </div>
      </div>
      {(error || clamp) && (
        <p id={msgId} className={`text-right text-[13px] ${error ? "font-semibold text-n-down" : "text-n-ink-2"}`} role={error ? "alert" : undefined}>
          {error ?? clamp}
        </p>
      )}
      {note && (
        <div id={noteId} className="text-[12.5px] leading-snug text-n-ink-2">
          {note}
        </div>
      )}
    </div>
  );
}

/** A plain text input on the same ruled line. */
export function TextIn({ id, label, value, onChange, testId, labelHidden = false }: { id?: string; label: ReactNode; value: string; onChange: (v: string) => void; testId?: string; labelHidden?: boolean }) {
  const uid = useId();
  const iid = id ?? uid;
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <label htmlFor={iid} className={labelHidden ? "sr-only" : "text-[14px] leading-snug text-n-ink-2"}>
        {label}
      </label>
      <input id={iid} type="text" value={value} maxLength={60} onChange={(e) => onChange(e.target.value)} data-testid={testId} className="min-h-11 w-full rounded-[2px] border-b border-n-line-strong bg-transparent py-1 text-[16px] text-n-ink outline-none outline-offset-2 focus:border-n-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--nx-accent)]" />
    </div>
  );
}

/**
 * One of a few named choices (radio buttons drawn as square-shouldered tiles). `value` may be null: no
 * choice is made for the user. Each option can carry a one-line meaning.
 */
export function Choice<T extends string>({
  name,
  legend,
  value,
  options,
  onChange,
  testId,
  columns = "sm:grid-cols-2",
  firstId,
}: {
  name: string;
  legend: ReactNode;
  value: T | null;
  options: { value: T; label: string; hint?: string }[];
  onChange: (v: T) => void;
  testId?: string;
  columns?: string;
  firstId?: string;
}) {
  return (
    <fieldset className="min-w-0" data-testid={testId}>
      <legend className="mb-2 text-[14px] text-n-ink-2">{legend}</legend>
      <div className={`grid grid-cols-1 gap-px overflow-hidden rounded-[4px] border border-n-line bg-n-line ${columns}`}>
        {options.map((o, i) => {
          const on = value === o.value;
          return (
            <label key={o.value} className={`nx-tint relative flex min-h-12 cursor-pointer flex-col justify-center gap-0.5 px-3.5 py-2.5 ${on ? "bg-n-surface-3" : "bg-n-surface"}`} data-testid={testId ? `${testId}-${o.value}` : undefined} data-on={on || undefined}>
              <input id={i === 0 ? firstId : undefined} type="radio" name={name} value={o.value} checked={on} onChange={() => onChange(o.value)} className="nx-focus peer sr-only" />
              <span aria-hidden className={`absolute inset-y-0 left-0 w-[3px] ${on ? "bg-n-accent" : "bg-transparent"}`} />
              <span className="flex items-center gap-2 text-[15px] font-medium text-n-ink">
                <span aria-hidden className={`grid h-4 w-4 shrink-0 place-items-center rounded-full border ${on ? "border-n-accent" : "border-n-ink-3"}`}>{on && <span className="h-2 w-2 rounded-full bg-n-accent" />}</span>
                {o.label}
              </span>
              {o.hint && <span className="pl-6 text-[12.5px] leading-snug text-n-ink-2">{o.hint}</span>}
              <span aria-hidden className="pointer-events-none absolute inset-0 rounded-[2px] peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:-outline-offset-2 peer-focus-visible:outline-n-accent" />
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

/** A plain-text action on the instrument (square shoulders, one hairline). */
export function Action({ onClick, children, testId, primary = false, disabled = false, type = "button", ariaExpanded, ariaControls }: { onClick?: () => void; children: ReactNode; testId?: string; primary?: boolean; disabled?: boolean; type?: "button" | "submit"; ariaExpanded?: boolean; ariaControls?: string }) {
  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      aria-expanded={ariaExpanded}
      aria-controls={ariaControls}
      className={`nx-press nx-focus inline-flex min-h-11 items-center justify-center gap-2 rounded-[3px] px-4 text-[14.5px] font-medium disabled:opacity-50 ${primary ? "bg-n-ink text-n-bg" : "border border-n-line-strong text-n-ink"}`}
      data-testid={testId}
    >
      {children}
    </button>
  );
}
