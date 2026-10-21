"use client";

import { useId, useState, type ReactNode } from "react";
import { parseField, type LimitKind } from "@/lib/tripEconomics";
import { Icon } from "@/v4/ui/Icon";

/**
 * Form fields for v4 reading surfaces: 16 px text (no zoom on iOS), at least 48 px tall, a visible
 * label, the unit inside the field, and an error or note tied to it (aria-describedby).
 */

const fieldBox = "nx-focus min-h-12 w-full rounded-xl border bg-n-surface text-[16px] text-n-ink";

/**
 * A number in the field's display unit. `value` is canonical; `scale` converts it for display
 * (display = value × scale), so pounds can be typed as kg or tons and $/lb as $/ton.
 */
export function NumberField({
  label,
  value,
  onChange,
  kind,
  unit,
  scale = 1,
  decimals = 2,
  hint,
  testId,
  placeholder,
  disabled,
  step,
  min = 0,
}: {
  label: ReactNode;
  value: number | null;
  onChange: (v: number | null) => void;
  kind: LimitKind;
  unit?: string;
  scale?: number;
  decimals?: number;
  hint?: ReactNode;
  testId?: string;
  placeholder?: string;
  disabled?: boolean;
  /** v7: a stepper (± this much, in the display unit); a press is the reader's own entry, never a default */
  step?: number;
  min?: number;
}) {
  const id = useId();
  const shown = value === null ? "" : String(Math.round(value * scale * 10 ** decimals) / 10 ** decimals);
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const describedBy = [error || note ? `${id}-msg` : null, hint ? `${id}-hint` : null].filter(Boolean).join(" ") || undefined;
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <label htmlFor={id} className="text-[14px] font-semibold leading-snug text-n-ink-2">
        {label}
      </label>
      <div className={step ? "nx-step" : "relative"}>
        {step && (
          <button type="button" tabIndex={-1} disabled={disabled || value === null} aria-label={`Less ${typeof label === "string" ? label : ""}`.trim()} onClick={() => value !== null && onChange(Math.max(min, Math.round((value * scale - step) * 1e6) / 1e6) / scale)} data-testid={testId ? `${testId}-dec` : undefined}>
            <Icon name="minus" className="h-4 w-4" />
          </button>
        )}
        <input
          id={id}
          type="text"
          inputMode="decimal"
          autoComplete="off"
          enterKeyHint="next"
          disabled={disabled}
          placeholder={placeholder}
          value={text ?? shown}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          data-testid={testId}
          onFocus={() => setText(shown)}
          onChange={(e) => {
            const t = e.target.value;
            setText(t);
            const p = parseField(t, kind);
            setError(p.error);
            setNote(null);
            if (p.error) return;
            if (p.value === null) onChange(null);
            else {
              const canonical = p.value / scale;
              onChange(canonical);
              if (p.clamped) setNote(`Limited to ${p.value.toLocaleString("en-US")}`);
            }
          }}
          onBlur={() => setText(null)}
          className={step ? "min-h-11 px-1 disabled:opacity-60" : `${fieldBox} nx-data px-3 ${unit ? "pr-16" : ""} ${error ? "border-n-down" : "border-n-line-strong"} disabled:opacity-60`}
        />
        {step && (
          <button type="button" tabIndex={-1} disabled={disabled} aria-label={`More ${typeof label === "string" ? label : ""}`.trim()} onClick={() => onChange(Math.round(((value ?? 0) * scale + step) * 1e6) / 1e6 / scale)} data-testid={testId ? `${testId}-inc` : undefined}>
            <Icon name="plus" className="h-4 w-4" />
          </button>
        )}
        {unit && !step && (
          <span aria-hidden className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-[14px] text-n-ink-3">
            {unit}
          </span>
        )}
      </div>
      {step && unit && <span className="-mt-0.5 text-center text-[12.5px] text-n-ink-3">{unit}</span>}
      {(error || note) && (
        <p id={`${id}-msg`} className={`text-[13.5px] ${error ? "font-semibold text-n-down" : "text-n-ink-3"}`} role={error ? "alert" : undefined}>
          {error ?? note}
        </p>
      )}
      {hint && (
        <div id={`${id}-hint`} className="text-[13px] leading-snug text-n-ink-3">
          {hint}
        </div>
      )}
    </div>
  );
}

export function TextField({ label, value, onChange, testId, maxLength = 60 }: { label: ReactNode; value: string; onChange: (v: string) => void; testId?: string; maxLength?: number }) {
  const id = useId();
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <label htmlFor={id} className="text-[14px] font-semibold leading-snug text-n-ink-2">
        {label}
      </label>
      <input id={id} type="text" value={value} maxLength={maxLength} onChange={(e) => onChange(e.target.value)} data-testid={testId} className={`${fieldBox} border-n-line-strong px-3`} />
    </div>
  );
}

export function Select<T extends string>({ label, value, onChange, options, testId, hint }: { label: ReactNode; value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; testId?: string; hint?: ReactNode }) {
  const id = useId();
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <label htmlFor={id} className="text-[14px] font-semibold leading-snug text-n-ink-2">
        {label}
      </label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value as T)} data-testid={testId} aria-describedby={hint ? `${id}-hint` : undefined} className={`${fieldBox} nx-select border-n-line-strong pl-3 font-semibold`}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      {hint && (
        <div id={`${id}-hint`} className="text-[13px] leading-snug text-n-ink-3">
          {hint}
        </div>
      )}
    </div>
  );
}

/** A checkbox row at least 44 px tall. */
export function Check({ label, checked, onChange, testId }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; testId?: string }) {
  return (
    <label className="flex min-h-11 cursor-pointer items-center gap-3 text-[15px] text-n-ink">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} data-testid={testId} className="nx-focus h-5 w-5 shrink-0 accent-[var(--nx-accent)]" />
      <span className="min-w-0">{label}</span>
    </label>
  );
}
