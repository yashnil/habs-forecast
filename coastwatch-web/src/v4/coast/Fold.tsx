"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Icon } from "@/v4/ui/Icon";

/**
 * A briefing note (v7): one folded row (title, count, one line saying what it holds) with its body
 * always mounted, so every word is in the page and one tap away. It opens by height (.nx-fold) and is
 * `hidden` again once closed. A link to any id inside it (#sea-h, #worth-h…) opens it first.
 * `inline` (v10 pass 2): a small "Season details ⌄" control set where its subject is, not a row in a
 * stack of accordions.
 */
export function Fold({
  id,
  title,
  count,
  summary,
  children,
  open: controlled,
  onOpenChange,
  inline = false,
  className = "",
  openFrom,
}: {
  id: string;
  title: ReactNode;
  count?: number | null;
  summary?: ReactNode;
  children: ReactNode;
  open?: boolean;
  onOpenChange?: (o: boolean) => void;
  inline?: boolean;
  className?: string;
  /** "md": folded on a phone only; from 768 px the body is always shown and the toggle is not (CSS only, no shift) */
  openFrom?: "md";
}) {
  const [own, setOwn] = useState(false);
  const open = controlled ?? own;
  const setOpen = useCallback(
    (o: boolean) => {
      if (controlled === undefined) setOwn(o);
      onOpenChange?.(o);
    },
    [controlled, onOpenChange],
  );
  const [bodyHidden, setBodyHidden] = useState(!open);
  const [expanded, setExpanded] = useState(open);
  useEffect(() => {
    if (open) {
      setBodyHidden(false);
      const t = window.setTimeout(() => setExpanded(true), 16);
      return () => window.clearTimeout(t);
    }
    setExpanded(false);
    const t = window.setTimeout(() => setBodyHidden(true), 240);
    return () => window.clearTimeout(t);
  }, [open]);
  const box = useRef<HTMLElement | null>(null);
  const openRef = useRef(setOpen);
  useEffect(() => {
    openRef.current = setOpen;
  }, [setOpen]);
  useEffect(() => {
    const inside = (hash: string) => {
      if (!hash || hash.length < 2) return false;
      const t = document.getElementById(decodeURIComponent(hash.slice(1)));
      return !!t && !!box.current?.contains(t);
    };
    const onHash = () => inside(window.location.hash) && openRef.current(true);
    const onClick = (e: MouseEvent) => {
      const a = (e.target as Element | null)?.closest?.('a[href^="#"]');
      if (a && inside(a.getAttribute("href") ?? "")) openRef.current(true);
    };
    onHash();
    window.addEventListener("hashchange", onHash);
    document.addEventListener("click", onClick, true);
    return () => {
      window.removeEventListener("hashchange", onHash);
      document.removeEventListener("click", onClick, true);
    };
  }, []);
  if (inline)
    return (
      <section ref={box} aria-labelledby={`${id}-h`} className={`scroll-mt-4 ${className}`} data-testid={`brief-${id}`} data-open={open}>
        <h3 id={`${id}-h`} className={`scroll-mt-4 ${openFrom ? "md:sr-only" : ""}`}>
          <button type="button" aria-expanded={open} aria-controls={`${id}-body`} onClick={() => setOpen(!open)} className="nx-focus group inline-flex min-h-11 items-center gap-1.5 rounded-[2px] text-left text-[14px] text-n-ink sm:min-h-9" data-testid={`brief-${id}-toggle`}>
            <span className="underline decoration-n-line-strong underline-offset-4 group-hover:decoration-n-ink">{title}</span>
            {count != null && <span className="nx-num text-[13px] text-n-ink-3">{count}</span>}
            {summary && <span className="text-[13px] text-n-ink-2 max-sm:hidden">{summary}</span>}
            <Icon name="chevronDown" className={`h-4 w-4 shrink-0 text-n-ink-3 transition-transform duration-[240ms] ease-[var(--nx-ease-move)] ${open ? "rotate-180" : ""}`} />
          </button>
        </h3>
        <div className={`nx-fold ${openFrom ? "md:!grid-rows-[1fr] md:[&>*]:!opacity-100" : ""}`} data-open={expanded}>
          <div id={`${id}-body`} hidden={openFrom ? undefined : bodyHidden} className={`flex flex-col gap-4 pb-4 pt-2 ${openFrom ? `md:pt-0 ${bodyHidden ? "max-md:hidden" : ""}` : ""}`}>
            {children}
          </div>
        </div>
      </section>
    );
  return (
    <section ref={box} aria-labelledby={`${id}-h`} className={`scroll-mt-4 border-t border-n-line ${className}`} data-testid={`brief-${id}`} data-open={open}>
      <h2 id={`${id}-h`} className="scroll-mt-4">
        <button type="button" aria-expanded={open} aria-controls={`${id}-body`} onClick={() => setOpen(!open)} className="nx-tint nx-focus group flex min-h-[60px] w-full items-center gap-3 rounded-lg py-3 text-left" data-testid={`brief-${id}-toggle`}>
          <span className="min-w-0 flex-1">
            <span className="flex items-baseline gap-2 text-[16.5px] font-semibold leading-6 text-n-ink">
              {title}
              {count != null && <span className="nx-num text-[15px] font-medium text-n-ink-3">{count}</span>}
            </span>
            {summary && <span className={`block text-[13.5px] font-normal leading-5 text-n-ink-2 ${open ? "" : "line-clamp-2 sm:line-clamp-1"}`}>{summary}</span>}
          </span>
          <Icon name="chevronDown" className={`h-[18px] w-[18px] shrink-0 text-n-ink-3 transition-transform duration-[240ms] ease-[var(--nx-ease-move)] ${open ? "rotate-180" : ""}`} />
        </button>
      </h2>
      <div className="nx-fold" data-open={expanded}>
        <div id={`${id}-body`} hidden={bodyHidden} className="flex flex-col gap-4 pb-6 pt-1">
          {children}
        </div>
      </div>
    </section>
  );
}
