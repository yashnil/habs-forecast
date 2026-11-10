"use client";

import { useEffect, useState, ViewTransition } from "react";
import { usePathname } from "next/navigation";

/**
 * The margin column (README §6): ruled, no container. Desktop and landscape: a column beside the sheet. Phone:
 * a sheet over the chart's lower part that rests at a detent showing the page's statement and opens to read
 * the rest (ds MarginSheet, on vaul, replaces this minimal one at integration). Its contents cross-fade with
 * React's <ViewTransition> when the route changes; the chart is never in the snapshot (shell.css keeps the
 * root out of it), so it keeps panning live underneath.
 */
export function MarginFrame({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  // a new page rests at the detent again
  useEffect(() => setOpen(false), [pathname]);
  return (
    <main id="nx-margin" className="n12-margin" data-nx-margin data-open={open ? "1" : "0"} tabIndex={-1} data-testid="nx-margin">
      <button type="button" className="n12-margin-handle" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="n12-margin-grip" aria-hidden="true" />
        <span className="n12-sr">{open ? "Show more of the chart" : "Read more"}</span>
      </button>
      <div className="n12-margin-scroll">
        <ViewTransition name="nx-margin" default="nx-margin-swap">
          <div className="n12-margin-page">{children}</div>
        </ViewTransition>
      </div>
    </main>
  );
}

/** A destination's margin until its feature lands: what this view is, in one sentence (journeys.md "Help"). */
export function PlaceholderMargin({ kicker, title, sentence, children, testid }: { kicker?: string; title: React.ReactNode; sentence: string; children?: React.ReactNode; testid?: string }) {
  return (
    <section className="nx-placeholder" data-testid={testid}>
      {kicker ? <p className="n12-kicker">{kicker}</p> : null}
      <h1 className="n12-statement n12-statement-m">{title}</h1>
      <p className="n12-lead">{sentence}</p>
      {children}
    </section>
  );
}
