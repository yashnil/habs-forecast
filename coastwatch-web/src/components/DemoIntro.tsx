"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { DEMO_COPY } from "@/content/copy";
import { Icon } from "@/components/ui/Icon";
import { GAP_LINKS, gapSentence } from "@/components/RegulatoryGap";
import type { OfficialDataset } from "@/generated/official";

/**
 * Portfolio-preview introduction (NEXT_PUBLIC_CW_DEMO=1 builds only): a short first-visit
 * card that says what CoastWatch is, what each map layer is, and what it is not. It opens
 * once per browser and again from the masthead "About" button.
 */
const SEEN_KEY = "cw-demo-intro-seen-v1";
const OPEN_EVENT = "cw:demo-about";

export function openDemoIntro() {
  window.dispatchEvent(new Event(OPEN_EVENT));
}

export function DemoAboutButton({ className = "", children }: { className?: string; children?: React.ReactNode }) {
  return (
    <button type="button" data-testid="demo-about" aria-haspopup="dialog" onClick={openDemoIntro} className={className}>
      {children ?? "About"}
    </button>
  );
}

export function DemoIntro({ official }: { official: OfficialDataset | null }) {
  const [open, setOpen] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let seen = false;
    try {
      seen = window.localStorage.getItem(SEEN_KEY) === "1";
    } catch {}
    if (!seen) setOpen(true);
    const onOpen = () => setOpen(true);
    window.addEventListener(OPEN_EVENT, onOpen);
    return () => window.removeEventListener(OPEN_EVENT, onOpen);
  }, []);

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  function close() {
    setOpen(false);
    try {
      window.localStorage.setItem(SEEN_KEY, "1");
    } catch {}
  }

  if (!open) return null;
  const c = DEMO_COPY;
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-navy-950/55 p-0 sm:items-center sm:p-6" onClick={close}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="demo-intro-h"
        data-testid="demo-intro"
        onClick={(e) => e.stopPropagation()}
        className="flex max-h-[88dvh] w-full max-w-full flex-col overflow-hidden rounded-t-2xl bg-surface text-ink shadow-2xl sm:max-w-[620px] sm:rounded-2xl"
        style={{ colorScheme: "light" }}
      >
        <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-5 pb-4 pt-5 sm:px-8 sm:pt-7">
          <div className="flex items-start justify-between gap-4">
            <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-accent">{c.eyebrow}</p>
            <button type="button" onClick={close} aria-label="Close introduction" className="-mr-1.5 -mt-1.5 rounded-md p-1.5 text-ink-3 hover:text-ink">
              <Icon name="close" />
            </button>
          </div>
          <h2 id="demo-intro-h" className="mt-1.5 font-display text-[25px] font-medium leading-[1.15] tracking-tight sm:text-[30px]">
            {c.heading}
          </h2>
          <p className="mt-3 text-[14px] leading-relaxed text-ink-2 sm:text-[15px]">{c.lede}</p>

          <ul className="mt-4 space-y-3 border-y sm:mt-5 border-hairline py-4">
            {c.layers.map((l) => (
              <li key={l.name} className="grid gap-1 text-[14px] leading-snug sm:grid-cols-[128px_1fr] sm:gap-3">
                <span>
                  <span
                    className={`inline-block rounded-full border px-2 py-0.5 text-[10.5px] font-semibold uppercase tracking-wider ${
                      l.kind === "model"
                        ? "border-model-line bg-model-bg text-model-ink"
                        : l.kind === "observation"
                          ? "border-measured-line bg-measured-bg text-measured"
                          : "border-official-line bg-official-bg text-official-ink"
                    }`}
                  >
                    {l.badge}
                  </span>
                </span>
                <span>
                  <span className="font-semibold text-ink">{l.name}.</span> <span className="text-ink-2">{l.text}</span>
                  {l.kind === "official" && <span className="mt-1 block text-official-ink" data-testid="demo-del-norte">{gapSentence(official)}</span>}
                </span>
              </li>
            ))}
          </ul>

          <p className="mt-4 rounded-lg bg-surface-3 px-3.5 py-3 text-[13px] leading-relaxed text-ink-2" data-testid="demo-caveat">
            {c.caveat} check{" "}
            <a className="font-medium text-accent underline underline-offset-2" href={GAP_LINKS[0].href} target="_blank" rel="noreferrer">
              CDFW
            </a>{" "}
            and{" "}
            <a className="font-medium text-accent underline underline-offset-2" href={GAP_LINKS[2].href} target="_blank" rel="noreferrer">
              CDPH
            </a>
            .
          </p>

        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-x-5 gap-y-2 border-t border-hairline px-5 py-3.5 sm:px-8">
            <button ref={closeRef} type="button" onClick={close} data-testid="demo-intro-start" className="rounded-lg bg-navy-950 px-5 py-3 text-[15px] sm:py-2.5 font-medium text-white hover:bg-navy-800">
              {c.cta}
            </button>
            <Link href="/sources" onClick={close} className="text-[14px] font-medium text-accent hover:underline">
              Data sources and status
            </Link>
            <a href="https://github.com/yashnil/habs-forecast" target="_blank" rel="noreferrer" className="text-[14px] font-medium text-accent hover:underline">
              Source code
            </a>
        </div>
      </div>
    </div>
  );
}
