"use client";
import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { Icon } from "./Icon";
import { Dialog, type DialogProps } from "./Overlay";
import { cx } from "./util";
import s from "./controls.module.css";

/* ---------------------------------------------------------------- Drawer */

export type DrawerProps = Omit<DialogProps, "placement">;

/**
 * A drawer of detail that leaves the page where it is: from the right on a wide screen, from the bottom on a
 * phone (saved trips, an alert's history). A dialog underneath: focus is held and returned, Escape closes.
 */
export function Drawer(props: DrawerProps) {
  return <Dialog {...props} placement="side" />;
}

/* ---------------------------------------------------------------- EditableText */

export interface EditableTextProps {
  /** What is being named: "Trip name". The accessible name of the field and of its edit control. */
  label: string;
  value: string;
  /** Called with the trimmed text when the reader commits (Enter or leaving the field). Never with an empty name. */
  onChange: (v: string) => void;
  /** Shown, quieter, when the value is empty. */
  placeholder?: string;
  maxLength?: number;
  /** head: a heading-sized name; body: in a line of text. */
  size?: "head" | "body";
  className?: string;
  "data-testid"?: string;
}

/**
 * A name edited in place: it reads as text with an edit control beside it; pressed, it is a text field with
 * the name selected. Enter or leaving the field keeps the change, Escape puts the old name back, and an empty
 * name is never kept.
 */
export function EditableText({ label, value, onChange, placeholder = "Untitled", maxLength = 60, size = "head", className, ...rest }: EditableTextProps) {
  const id = useId();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const input = useRef<HTMLInputElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);
  const begin = () => {
    setDraft(value);
    setEditing(true);
  };
  const finish = (keep: boolean, refocus: boolean) => {
    const next = draft.trim();
    setEditing(false);
    if (keep && next && next !== value) onChange(next);
    if (refocus) requestAnimationFrame(() => button.current?.focus());
  };
  const key = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      finish(true, true);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      finish(false, true);
    }
  };
  return (
    <span className={cx(s.editable, className)} data-size={size} data-editing={editing || undefined} data-testid={rest["data-testid"]}>
      {editing ? (
        <>
          <label htmlFor={id} className={s.vh}>
            {label}
          </label>
          <input id={id} ref={input} className={s.editInput} value={draft} maxLength={maxLength} onChange={(e) => setDraft(e.target.value)} onKeyDown={key} onBlur={() => finish(true, false)} autoComplete="off" />
        </>
      ) : (
        <>
          <span className={cx(s.editText, !value && s.editEmpty)}>{value || placeholder}</span>
          <button ref={button} type="button" className={s.editBtn} onClick={begin} aria-label={`Rename: ${label}`}>
            Rename
          </button>
        </>
      )}
    </span>
  );
}

/* ---------------------------------------------------------------- OptionList */

export interface OptionItem<T extends string> {
  value: T;
  label: ReactNode;
  /** A second line: what is known about it ("12 of 49 samples · latest Mar 3, 2024"). */
  meta?: ReactNode;
  /** At the row's end: a mark, a count, a small figure. */
  end?: ReactNode;
  /** Across the row's whole width, under its words: a thin strip (when this variable was measured). Decorative: say the same in `meta`. */
  strip?: ReactNode;
  disabled?: boolean;
}
export interface OptionListProps<T extends string> {
  /** Accessible name: "Variable". */
  label: string;
  hideLabel?: boolean;
  options: OptionItem<T>[];
  value: T | null;
  onChange: (v: T) => void;
  className?: string;
}

/**
 * One choice from a list where each choice has something to say for itself (the variable rail: each variable
 * with how much of it there is and when it was last measured). A radio group: Up and Down move and choose; the
 * chosen row carries an ink bar on its leading edge. A choice with nothing behind it stays in the list, disabled
 * and saying why in its second line, never silently missing.
 */
export function OptionList<T extends string>({ label, hideLabel, options, value, onChange, className }: OptionListProps<T>) {
  const id = useId();
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const at = options.findIndex((o) => o.value === value);
  const first = options.findIndex((o) => !o.disabled);
  const key = (i: number) => (e: KeyboardEvent<HTMLButtonElement>) => {
    const d = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    for (let n = 1; n <= options.length; n++) {
      const j = (i + d * n + options.length * n) % options.length;
      if (!options[j].disabled) {
        onChange(options[j].value);
        refs.current[j]?.focus();
        return;
      }
    }
  };
  return (
    <div className={cx(s.optWrap, className)}>
      <span id={id} className={cx(s.optLabel, hideLabel && s.vh)}>
        {label}
      </span>
      <div role="radiogroup" aria-labelledby={id} className={s.optList}>
        {options.map((o, i) => (
          <button
            key={o.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={o.value === value}
            aria-disabled={o.disabled || undefined}
            tabIndex={o.value === value || (at === -1 && i === first) ? 0 : -1}
            className={s.opt}
            onClick={() => !o.disabled && onChange(o.value)}
            onKeyDown={key(i)}
            data-testid={`option-${o.value}`}
          >
            <span className={s.optRow}>
              <span className={s.optText}>
                <span className={s.optName}>{o.label}</span>
                {o.meta ? <span className={s.optMeta}>{o.meta}</span> : null}
              </span>
              {o.end ? <span className={s.optEnd}>{o.end}</span> : null}
            </span>
            {o.strip ? (
              <span className={s.optStrip} aria-hidden="true">
                {o.strip}
              </span>
            ) : null}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- Inspector */

export interface InspectorProps {
  /** What is inspected: "Sample, Mar 3, 2024". Also the region's accessible name. */
  title: ReactNode;
  /** Under the title: where and what ("Monterey Wharf · pDA"). */
  subtitle?: ReactNode;
  onClose?: () => void;
  /** Links for this sample (source, QC notes), at the foot. */
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
  "data-testid"?: string;
}

/**
 * The detail card for one inspected thing (a pinned sample). On a wide screen it is a ruled card where it
 * stands, beside or under the plot. On a phone it is pinned above the destinations, at most half the screen
 * tall and scrolling inside itself, so the plot above it stays in view while the reader steps through samples.
 * Not a dialog: the page stays live around it. Changes are announced politely.
 */
export function Inspector({ title, subtitle, onClose, actions, className, children, ...rest }: InspectorProps) {
  return (
    <section className={cx(s.inspector, className)} aria-label={typeof title === "string" ? title : "Details"} aria-live="polite" data-nx-inspector data-testid={rest["data-testid"]}>
      <header className={s.inspHead}>
        <div>
          <h3 className={s.inspTitle}>{title}</h3>
          {subtitle ? <p className={s.inspSub}>{subtitle}</p> : null}
        </div>
        {onClose ? (
          <button type="button" className={s.hintClose} onClick={onClose} aria-label="Close the details">
            <Icon name="close" />
          </button>
        ) : null}
      </header>
      <div className={s.inspBody}>{children}</div>
      {actions ? <div className={s.inspActions}>{actions}</div> : null}
    </section>
  );
}

/** Label and value pairs inside an Inspector (value, units, source, date, place, QC): a definition list, ruled. */
export function Facts({ items, className }: { items: Array<{ label: ReactNode; value: ReactNode }>; className?: string }) {
  return (
    <dl className={cx(s.factsList, className)}>
      {items.map((f, i) => (
        <div key={i} className={s.factRow}>
          <dt>{f.label}</dt>
          <dd>{f.value}</dd>
        </div>
      ))}
    </dl>
  );
}
