"use client";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useId, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { Icon } from "./Icon";
import { press, reveal, select } from "./motion";
import { cx } from "./util";
import s from "./controls.module.css";

/* ---------------------------------------------------------------- RangeControl */

export interface RangeControlProps<T extends string> {
  /** Accessible name: "Window". */
  label: string;
  presets: Array<{ value: T; label: string }>;
  /** The chosen preset, or null when the window is the reader's own (a brushed range). */
  value: T | null;
  onChange: (v: T) => void;
  /** Words shown as the chosen item when `value` is null ("Custom"). */
  customLabel?: string;
  /** The "latest" action: moves the window to the latest real observation. Its label says the date. */
  onJumpLatest?: () => void;
  latestLabel?: ReactNode;
  /** Anything else for the same row (a "Show as table" control). */
  action?: ReactNode;
  className?: string;
}

/**
 * A time range as quiet words with an underline that slides to the chosen one (`select`): 90 d, 1 y, 3 y, All,
 * and the reader's own range when the brush made one. Beside it, a slot for the "latest sample" action, so the
 * latest real observation is one press away and never a guess at a range.
 */
export function RangeControl<T extends string>({ label, presets, value, onChange, customLabel = "Custom", onJumpLatest, latestLabel = "Latest sample", action, className }: RangeControlProps<T>) {
  const id = useId();
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const at = presets.findIndex((p) => p.value === value);
  const key = (i: number) => (e: KeyboardEvent<HTMLButtonElement>) => {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const n = (i + d + presets.length) % presets.length;
    onChange(presets[n].value);
    refs.current[n]?.focus();
  };
  return (
    <div className={cx(s.range, className)}>
      <div className={s.rangeSet} role="radiogroup" aria-label={label}>
        {presets.map((p, i) => (
          <button
            key={p.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={p.value === value}
            tabIndex={p.value === value || (at === -1 && i === 0) ? 0 : -1}
            className={s.rangeItem}
            onClick={() => onChange(p.value)}
            onKeyDown={key(i)}
            data-testid={`range-${p.value}`}
          >
            {p.label}
            {p.value === value ? <motion.span layoutId={`${id}-u`} className={s.rangeOn} transition={select} aria-hidden /> : null}
          </button>
        ))}
        {at === -1 ? (
          <span className={cx(s.rangeItem, s.rangeCustom)} data-testid="range-custom">
            {customLabel}
            <motion.span layoutId={`${id}-u`} className={s.rangeOn} transition={select} aria-hidden />
          </span>
        ) : null}
      </div>
      {onJumpLatest ? (
        <button type="button" className={s.rangeLatest} onClick={onJumpLatest} data-testid="range-latest">
          {latestLabel}
        </button>
      ) : null}
      {action}
    </div>
  );
}

/* ---------------------------------------------------------------- ScrubNumber */

export interface ScrubNumberProps {
  label: string;
  /** The figure as the reader typed it ("" when empty: nothing is assumed). */
  value: string;
  onChange: (v: string) => void;
  /** One step of the arrow keys and of 8 px of drag (Shift: ten steps). */
  step?: number;
  min?: number;
  max?: number;
  prefix?: string;
  unit?: string;
  /** Decimals kept when stepping (default: those of `step`). */
  decimals?: number;
  hint?: ReactNode;
  error?: string;
  disabled?: boolean;
  /** Visible width of the figure, in characters. */
  size?: number;
  className?: string;
  "data-testid"?: string;
}

const decimalsOf = (n: number) => (String(n).split(".")[1] ?? "").length;

/**
 * A number the reader types, steps with the arrow keys, or drags sideways on its label (a scrub), so a change
 * of input shows in the result as the hand moves. It is a native text input underneath: typing, paste and the
 * numeric keypad work as anywhere. Empty stays empty; a drag starts from 0 only once the reader drags.
 */
export function ScrubNumber({ label, value, onChange, step = 1, min = 0, max = Number.MAX_SAFE_INTEGER, prefix, unit, decimals, hint, error, disabled, size = 6, className, ...rest }: ScrubNumberProps) {
  const id = useId();
  const dp = decimals ?? decimalsOf(step);
  const drag = useRef<{ x: number; from: number } | null>(null);
  const [scrubbing, setScrubbing] = useState(false);
  const num = () => {
    const n = parseFloat(value.replace(/,/g, ""));
    return Number.isFinite(n) ? n : 0;
  };
  const put = (n: number) => onChange(Math.min(max, Math.max(min, n)).toFixed(dp));
  const key = (e: KeyboardEvent<HTMLInputElement>) => {
    const d = e.key === "ArrowUp" ? 1 : e.key === "ArrowDown" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    put(num() + d * step * (e.shiftKey ? 10 : 1));
  };
  const down = (e: ReactPointerEvent<HTMLLabelElement>) => {
    if (disabled || e.pointerType === "touch") return;
    drag.current = { x: e.clientX, from: num() };
  };
  const move = (e: ReactPointerEvent<HTMLLabelElement>) => {
    const d = drag.current;
    if (!d) return;
    const steps = Math.trunc((e.clientX - d.x) / 8);
    if (!scrubbing && Math.abs(e.clientX - d.x) < 4) return;
    if (!scrubbing) {
      setScrubbing(true);
      e.currentTarget.setPointerCapture(e.pointerId);
    }
    put(d.from + steps * step * (e.shiftKey ? 10 : 1));
  };
  const up = () => {
    drag.current = null;
    setScrubbing(false);
  };
  return (
    <div className={cx(s.scrub, disabled && s.isDisabled, className)} data-scrubbing={scrubbing || undefined}>
      <label htmlFor={id} className={s.scrubLabel} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
        {label}
      </label>
      <span className={s.scrubBox} data-invalid={error ? "" : undefined}>
        {prefix ? <span className={s.scrubUnit}>{prefix}</span> : null}
        <input
          id={id}
          className={s.scrubInput}
          type="text"
          inputMode={dp ? "decimal" : "numeric"}
          autoComplete="off"
          size={size}
          value={value}
          disabled={disabled}
          aria-invalid={error ? true : undefined}
          aria-describedby={hint || error ? `${id}-h` : undefined}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={key}
          data-testid={rest["data-testid"]}
        />
        {unit ? <span className={s.scrubUnit}>{unit}</span> : null}
      </span>
      {error || hint ? (
        <span id={`${id}-h`} className={cx(s.hint, error && s.hintError)}>
          {error ?? hint}
        </span>
      ) : null}
    </div>
  );
}

/* ---------------------------------------------------------------- Disclosure */

export interface DisclosureProps {
  /** The row's name: "Station". */
  summary: ReactNode;
  /** A short value shown on the row beside the name, closed or open. */
  value?: ReactNode;
  defaultOpen?: boolean;
  /** Controlled open state, with `onOpenChange`. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  id?: string;
  className?: string;
  children: ReactNode;
  "data-testid"?: string;
}

/**
 * A row that opens in place: its name and value on one line, its detail under it (`reveal`). The body stays in
 * the document while closed only as hidden content, so find-in-page and assistive tech are not misled.
 */
export function Disclosure({ summary, value, defaultOpen = false, open: controlled, onOpenChange, id, className, children, ...rest }: DisclosureProps) {
  const own = useId();
  const bodyId = id ?? own;
  const [mine, setMine] = useState(defaultOpen);
  const open = controlled ?? mine;
  const reduce = useReducedMotion();
  const set = (o: boolean) => {
    if (controlled === undefined) setMine(o);
    onOpenChange?.(o);
  };
  return (
    <div className={cx(s.disclosure, className)} data-open={open || undefined} data-testid={rest["data-testid"]}>
      <motion.button type="button" className={s.discHead} aria-expanded={open} aria-controls={bodyId} onClick={() => set(!open)} {...press}>
        <span className={s.discSummary}>{summary}</span>
        {value !== undefined ? <span className={s.discValue}>{value}</span> : null}
        <Icon name="chevron" size={20} direction={open ? "up" : "down"} className={s.discIcon} />
      </motion.button>
      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            id={bodyId}
            className={s.discBody}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={reduce ? { duration: 0 } : reveal}
          >
            <div className={s.discInner}>{children}</div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

/* ---------------------------------------------------------------- ToggleRow and Hint */

export interface ToggleRowProps {
  label: ReactNode;
  /** What turning it on does, in one plain line under the label. */
  hint?: ReactNode;
  checked: boolean;
  onChange: (v: boolean) => void;
  /** The state in words beside the switch, never a switch alone. */
  onWord?: string;
  offWord?: string;
  disabled?: boolean;
  className?: string;
  "data-testid"?: string;
}

/** A setting as one 44 px row: its name and what it does on the left, a switch and its state in words on the right. */
export function ToggleRow({ label, hint, checked, onChange, onWord = "On", offWord = "Off", disabled, className, ...rest }: ToggleRowProps) {
  const id = useId();
  return (
    <div className={cx(s.toggleRow, disabled && s.isDisabled, className)}>
      <span className={s.toggleText}>
        <span id={`${id}-l`} className={s.toggleLabel}>
          {label}
        </span>
        {hint ? (
          <span id={`${id}-h`} className={s.hint}>
            {hint}
          </span>
        ) : null}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-labelledby={`${id}-l`}
        aria-describedby={hint ? `${id}-h` : undefined}
        className={s.switch}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        data-testid={rest["data-testid"]}
      >
        <span className={s.track} data-on={checked || undefined}>
          <motion.span className={s.knob} layout transition={select} />
        </span>
        <span className={s.switchWord}>{checked ? onWord : offWord}</span>
      </button>
    </div>
  );
}

export interface HintProps {
  children: ReactNode;
  /** note (default): how something works; caution: what it does not do (ruled in ink, never a warning hue). */
  tone?: "note" | "caution";
  /** Dismissible first-use guidance: shows a close control. */
  onDismiss?: () => void;
  className?: string;
}

/** A line of help where it is needed (contextual first-use guidance, what an alert actually does): never a tutorial. */
export function Hint({ children, tone = "note", onDismiss, className }: HintProps) {
  return (
    <p className={cx(s.inlineHint, className)} data-tone={tone} role="note">
      <span>{children}</span>
      {onDismiss ? (
        <button type="button" className={s.hintClose} onClick={onDismiss} aria-label="Dismiss">
          <Icon name="close" size={20} />
        </button>
      ) : null}
    </p>
  );
}
