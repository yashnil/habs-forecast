"use client";

import { useCallback, useEffect, useId, useRef, useState, type ElementType, type ReactNode } from "react";

/**
 * Expand in place: a title that opens its body by height (grid rows, 240 ms). The words stay in the DOM when
 * folded (inert, so nothing in them takes focus while closed). A link to an id inside it opens it first.
 *
 * Use for: a detail layer under a sentence and a graphic.
 * Do not use for: navigation, or content the reader needs to compare while another fold is open (that is a Sheet).
 */
export function Fold({
  title,
  children,
  open: controlled,
  defaultOpen = false,
  onOpenChange,
  summary,
  heading: H = "h3",
  id,
  className = "",
  triggerClassName = "",
  bodyClassName = "",
  testid,
}: {
  title: ReactNode;
  children: ReactNode;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** a short line after the title, kept while folded */
  summary?: ReactNode;
  heading?: ElementType;
  id?: string;
  className?: string;
  triggerClassName?: string;
  bodyClassName?: string;
  testid?: string;
}) {
  const auto = useId();
  const base = id ?? `fold-${auto}`;
  const [own, setOwn] = useState(defaultOpen);
  const open = controlled ?? own;
  const set = useCallback(
    (o: boolean) => {
      if (controlled === undefined) setOwn(o);
      onOpenChange?.(o);
    },
    [controlled, onOpenChange],
  );
  const box = useRef<HTMLDivElement | null>(null);
  const setRef = useRef(set);
  useEffect(() => {
    setRef.current = set;
  }, [set]);
  useEffect(() => {
    const inside = (hash: string) => {
      if (!hash || hash.length < 2) return false;
      const t = document.getElementById(decodeURIComponent(hash.slice(1)));
      return !!t && !!box.current?.contains(t);
    };
    const onHash = () => inside(window.location.hash) && setRef.current(true);
    const onClick = (e: MouseEvent) => {
      const a = (e.target as Element | null)?.closest?.('a[href^="#"]');
      if (a && inside(a.getAttribute("href") ?? "")) setRef.current(true);
    };
    onHash();
    window.addEventListener("hashchange", onHash);
    document.addEventListener("click", onClick, true);
    return () => {
      window.removeEventListener("hashchange", onHash);
      document.removeEventListener("click", onClick, true);
    };
  }, []);
  return (
    <div ref={box} className={className} data-open={open} data-testid={testid}>
      <H>
        <button type="button" aria-expanded={open} aria-controls={`${base}-body`} onClick={() => set(!open)} className={`nx-focus nx-m-fold-t ${triggerClassName}`}>
          <span className="min-w-0 flex-1">
            {title}
            {summary ? <span className="nx-m-fold-s"> {summary}</span> : null}
          </span>
          <svg aria-hidden viewBox="0 0 24 24" className="nx-m-fold-c" data-open={open} fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
            <path d="m6 9 6 6 6-6" />
          </svg>
        </button>
      </H>
      <div className="nx-fold" data-open={open}>
        <div id={`${base}-body`} inert={!open} className={bodyClassName}>
          {children}
        </div>
      </div>
    </div>
  );
}
