"use client";

import { useEffect, useState } from "react";
import { Micro } from "@/nx/ds/synth/Grammar";
import s from "./info.module.css";

export type ContentsItem = { id: string; label: string };

/**
 * A long page's contents: its sections, beside the text on a wide sheet (held in view while the page scrolls) and over
 * it on a narrow one. The section being read is marked (aria-current), from where the headings stand in the sheet.
 */
export function Contents({ items }: { items: ContentsItem[] }) {
  const [current, setCurrent] = useState<string | null>(null);
  useEffect(() => {
    const els = items.map((i) => document.getElementById(i.id)).filter((e): e is HTMLElement => !!e);
    if (!els.length || typeof IntersectionObserver === "undefined") return;
    const seen = new Map<string, boolean>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) seen.set(e.target.id, e.isIntersecting);
        // the first heading in the reading band; else keep the last one passed
        const first = els.find((el) => seen.get(el.id));
        if (first) setCurrent(first.id);
      },
      { rootMargin: "0px 0px -65% 0px" },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [items]);
  return (
    <nav className={s.toc} aria-label="On this page" data-testid="nx-info-contents">
      <Micro as="p">On this page</Micro>
      <ol>
        {items.map((i) => (
          <li key={i.id}>
            <a href={`#${i.id}`} aria-current={current === i.id ? "true" : undefined}>
              {i.label}
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
