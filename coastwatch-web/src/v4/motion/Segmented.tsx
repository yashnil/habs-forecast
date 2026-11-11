"use client";

import { useId, useLayoutEffect, useRef, type ReactNode, type RefObject } from "react";
import { prefersReducedMotion } from "./reduced";

const GLIDE_MS = 260;
const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const easeOut = (t: number) => 1 - Math.pow(1 - t, 4);

/**
 * One position glided from the last rendered one, in the box's own coordinates: where the thumb is now (the
 * running glide's own place if one is under way, so a quick second choice turns it around), toward `now`.
 */
export function glideFrom(prev: { x: number; w: number; fromX?: number; fromW?: number; at?: number } | null, nowMs: number, durationMs: number): { x: number; w: number } | null {
  if (!prev) return null;
  if (prev.at == null || prev.fromX == null || prev.fromW == null) return { x: prev.x, w: prev.w };
  const k = Math.min(1, (nowMs - prev.at) / durationMs);
  if (k >= 1) return { x: prev.x, w: prev.w };
  const e = easeOut(k);
  return { x: prev.fromX + (prev.x - prev.fromX) * e, w: prev.fromW + (prev.w - prev.fromW) * e };
}

/**
 * Slides a thumb to the new item when the choice changes (FLIP with WAAPI, 260 ms). The thumb is rendered
 * inside the active item (`thumb` selector), so it is right in the server HTML and without script. Instant
 * under reduced motion. `storageKey` carries its last place across page loads (TopBar's nav pill does this).
 *
 * Use for: a segmented control or tabs. Not for a day step or a layer switch's own content (those cut).
 */
export function useGlide(box: RefObject<HTMLElement | null>, key: string | number, { thumb = "[data-glide-thumb]", on = true, storageKey }: { thumb?: string; on?: boolean; storageKey?: string } = {}) {
  const last = useRef<{ x: number; w: number; fromX?: number; fromW?: number; at?: number } | null>(null);
  useLayoutEffect(() => {
    const root = box.current;
    const el = root?.querySelector<HTMLElement>(thumb);
    if (!on || !root || !el) return;
    const b = root.getBoundingClientRect();
    const p = el.getBoundingClientRect();
    const now = { x: p.left - b.left, w: p.width };
    let prev = last.current;
    if (!prev && storageKey) {
      try {
        prev = JSON.parse(sessionStorage.getItem(storageKey) ?? "null");
      } catch {
        prev = null;
      }
    }
    const at = performance.now();
    const from = glideFrom(prev, at, GLIDE_MS);
    last.current = { ...now };
    try {
      if (storageKey) sessionStorage.setItem(storageKey, JSON.stringify(now));
    } catch {
      // storage unavailable: the glide still works on this page
    }
    if (!from || (Math.abs(from.x - now.x) < 1 && Math.abs(from.w - now.w) < 1) || prefersReducedMotion() || now.w === 0) return;
    last.current = { ...now, fromX: from.x, fromW: from.w, at };
    el.style.transformOrigin = "left center";
    el.animate?.([{ transform: `translateX(${from.x - now.x}px) scaleX(${from.w / now.w})` }, { transform: "none" }], { duration: GLIDE_MS, easing: EASE });
    // on a new choice only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [box, key, on]);
}

export type SegmentedOption<V extends string> = { value: V; label: ReactNode; disabled?: boolean; testid?: string };

/**
 * A choice among two to five items, with a gliding thumb. `mode="tabs"` is a tablist (give each tab its
 * panel with `panelIdFor`); `mode="radio"` (default) picks a value. Arrows, Home and End move and choose.
 *
 * Use for: scenarios, a chart's window, a small set of views.
 * Do not use for: more than five items, or actions (a segment selects; it does not do).
 */
export function Segmented<V extends string>({
  value,
  onChange,
  options,
  label,
  mode = "radio",
  panelIdFor,
  className = "",
  testid,
}: {
  value: V;
  onChange: (v: V) => void;
  options: SegmentedOption<V>[];
  /** names the group for assistive tech */
  label: string;
  mode?: "radio" | "tabs";
  panelIdFor?: (v: V) => string;
  className?: string;
  testid?: string;
}) {
  const box = useRef<HTMLDivElement | null>(null);
  const name = useId();
  useGlide(box, value);
  const live = options.filter((o) => !o.disabled);
  const move = (dir: 1 | -1 | "home" | "end") => {
    const i = live.findIndex((o) => o.value === value);
    const n = live.length;
    const next = dir === "home" ? live[0] : dir === "end" ? live[n - 1] : live[(i + dir + n) % n];
    if (!next) return;
    onChange(next.value);
    box.current?.querySelector<HTMLElement>(`[data-seg="${CSS.escape(next.value)}"]`)?.focus();
  };
  return (
    <div
      ref={box}
      role={mode === "tabs" ? "tablist" : "radiogroup"}
      aria-label={label}
      className={`nx-seg ${className}`}
      data-testid={testid}
      onKeyDown={(e) => {
        const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : e.key === "Home" ? "home" : e.key === "End" ? "end" : null;
        if (d == null) return;
        e.preventDefault();
        move(d);
      }}
    >
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            id={mode === "tabs" ? `${name}-${o.value}` : undefined}
            role={mode === "tabs" ? "tab" : "radio"}
            aria-selected={mode === "tabs" ? on : undefined}
            aria-checked={mode === "radio" ? on : undefined}
            aria-controls={mode === "tabs" && panelIdFor ? panelIdFor(o.value) : undefined}
            tabIndex={on ? 0 : -1}
            disabled={o.disabled}
            data-seg={o.value}
            data-on={on || undefined}
            data-testid={o.testid}
            className="nx-seg-o nx-focus"
            onClick={() => onChange(o.value)}
          >
            {on ? <span data-glide-thumb aria-hidden className="nx-seg-thumb" /> : null}
            <span className="relative">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}
