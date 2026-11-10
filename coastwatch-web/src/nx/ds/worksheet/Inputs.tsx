"use client";
import { motion } from "motion/react";
import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";
import { Icon } from "../Icon";
import { press, settle } from "../motion";
import { cx } from "../util";
import s from "./worksheet.module.css";

/* ---------------- Field ---------------- */

export interface FieldProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  /** Unit before the figure ("$"). */
  prefix?: string;
  /** Unit after the figure ("lb", "% of gross", "gal"). */
  suffix?: string;
  inputMode?: "decimal" | "numeric";
  /** A sourced reference that enters only on a tap (T47). */
  reference?: { value: string; words: string; source: string };
  /** Set once the reader took the reference: the entry stays labelled. */
  fromReference?: string;
  hint?: ReactNode;
  error?: string;
  disabled?: boolean;
  /** Visible width of the figure, in characters. */
  size?: number;
  /** Inline: the label sits left, as on a ledger line. */
  layout?: "stacked" | "inline";
  hideLabel?: boolean;
  className?: string;
}

/** A figure with its unit. Nothing is pre-filled; a blank reads as blank, never as 0. */
export function Field({ label, value, onChange, prefix, suffix, inputMode = "decimal", reference, fromReference, hint, error, disabled, size = 6, layout = "stacked", hideLabel, className }: FieldProps) {
  const id = useId();
  const noteId = `${id}-note`;
  const described = hint || error || fromReference || reference ? noteId : undefined;
  return (
    <div className={cx(s.field, layout === "inline" && s.inline, error && s.hasError, disabled && s.isDisabled, className)}>
      <label htmlFor={id} className={cx(s.label, hideLabel && s.vh)}>
        {label}
      </label>
      <span className={s.figure}>
        {prefix ? <span className={s.unitBefore}>{prefix}</span> : null}
        <input
          id={id}
          className={s.input}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          inputMode={inputMode}
          autoComplete="off"
          spellCheck={false}
          disabled={disabled}
          aria-invalid={error ? true : undefined}
          aria-describedby={described}
          style={{ width: `${size + 1}ch` }}
        />
        {suffix ? <span className={s.unitAfter}>{suffix}</span> : null}
      </span>
      {described ? (
        <span id={noteId} className={s.note}>
          {error ? (
            <span className={s.error}>{error}</span>
          ) : fromReference ? (
            <span>Your entry, from {fromReference}.</span>
          ) : reference && !value ? (
            <button type="button" className={s.refButton} onClick={() => onChange(reference.value)}>
              Use {reference.words} ({reference.source})
            </button>
          ) : (
            hint
          )}
        </span>
      ) : null}
    </div>
  );
}

/* ---------------- Stepper ---------------- */

export interface StepperProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  disabled?: boolean;
  className?: string;
}

/** A whole-number input with − and + at the touch target. Arrow keys step; Home and End jump to the limits. */
export function Stepper({ label, value, onChange, min = 0, max = 99, step = 1, unit, disabled, className }: StepperProps) {
  const id = useId();
  const clamp = (v: number) => Math.min(max, Math.max(min, v));
  const key = (e: KeyboardEvent<HTMLInputElement>) => {
    const map: Record<string, number> = { ArrowUp: value + step, ArrowDown: value - step, PageUp: value + step * 10, PageDown: value - step * 10, Home: min, End: max };
    if (e.key in map) {
      e.preventDefault();
      onChange(clamp(map[e.key]));
    }
  };
  return (
    <div className={cx(s.stepper, disabled && s.isDisabled, className)} role="group" aria-labelledby={`${id}-l`}>
      <span id={`${id}-l`} className={s.label}>
        {label}
      </span>
      <span className={s.stepRow}>
        <motion.button type="button" className={s.stepBtn} onClick={() => onChange(clamp(value - step))} disabled={disabled || value <= min} aria-label={`Fewer: ${label}`} {...press}>
          <Icon name="minus" />
        </motion.button>
        <span className={s.stepValue}>
          <input
            id={id}
            className={s.stepInput}
            role="spinbutton"
            inputMode="numeric"
            value={String(value)}
            aria-valuenow={value}
            aria-valuemin={min}
            aria-valuemax={max}
            aria-labelledby={`${id}-l`}
            disabled={disabled}
            onKeyDown={key}
            onChange={(e) => {
              const n = parseInt(e.target.value.replace(/\D/g, ""), 10);
              if (Number.isFinite(n)) onChange(clamp(n));
            }}
          />
          {unit ? <span className={s.unitAfter}>{unit}</span> : null}
        </span>
        <motion.button type="button" className={s.stepBtn} onClick={() => onChange(clamp(value + step))} disabled={disabled || value >= max} aria-label={`More: ${label}`} {...press}>
          <Icon name="plus" />
        </motion.button>
      </span>
    </div>
  );
}

/* ---------------- Segmented ---------------- */

export interface SegmentedProps<T extends string> {
  label: string;
  options: Array<{ value: T; label: string }>;
  value: T | null;
  onChange: (v: T) => void;
  hideLabel?: boolean;
  disabled?: boolean;
  className?: string;
}

/**
 * One choice from a few, side by side (a radio group). Arrow keys move and choose. A null value is allowed
 * where nothing may be assumed (who pays fuel, E2): no segment is chosen until the reader chooses.
 */
export function Segmented<T extends string>({ label, options, value, onChange, hideLabel, disabled, className }: SegmentedProps<T>) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const at = options.findIndex((o) => o.value === value);
  const key = (i: number) => (e: KeyboardEvent<HTMLButtonElement>) => {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const n = (i + d + options.length) % options.length;
    onChange(options[n].value);
    refs.current[n]?.focus();
  };
  const id = useId();
  return (
    <div className={cx(s.segWrap, disabled && s.isDisabled, className)}>
      <span id={id} className={cx(s.label, hideLabel && s.vh)}>
        {label}
      </span>
      <div className={s.segmented} role="radiogroup" aria-labelledby={id}>
        {options.map((o, i) => (
          <motion.button
            key={o.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={o.value === value}
            tabIndex={o.value === value || (at === -1 && i === 0) ? 0 : -1}
            className={s.segment}
            onClick={() => onChange(o.value)}
            onKeyDown={key(i)}
            disabled={disabled}
            {...press}
          >
            {o.label}
          </motion.button>
        ))}
      </div>
    </div>
  );
}

/* ---------------- Toggle ---------------- */

export interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  /** The state in words, beside the switch: never a switch alone. */
  onWord?: string;
  offWord?: string;
  disabled?: boolean;
  className?: string;
}

/** A switch that says its state in words. */
export function Toggle({ label, checked, onChange, onWord = "On", offWord = "Off", disabled, className }: ToggleProps) {
  return (
    <motion.button
      type="button"
      role="switch"
      aria-checked={checked}
      className={cx(s.toggle, disabled && s.isDisabled, className)}
      onClick={() => onChange(!checked)}
      disabled={disabled}
      {...press}
    >
      <span className={s.toggleLabel}>{label}</span>
      <span className={s.toggleState}>
        <span className={s.track} data-on={checked || undefined}>
          <motion.span className={s.knob} layout transition={settle} />
        </span>
        <span className={s.toggleWord}>{checked ? onWord : offWord}</span>
      </span>
    </motion.button>
  );
}
