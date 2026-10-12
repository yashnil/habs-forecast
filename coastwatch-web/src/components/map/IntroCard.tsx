"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { KindChip } from "@/components/ui/Kinds";

import { TOUR_SEEN_KEY as KEY } from "./Tour";

/**
 * First visit only: what this map is, in two sentences, with the three kinds of information it
 * shows, and the start of the guided tour. Dismissed once (remembered on this device); never a
 * modal, never in front of official information. Hidden until mounted so it never flashes for
 * returning visitors. The tour itself can be reopened at any time from the map's Tour control.
 */
export function IntroCard({ className = "", onTour }: { className?: string; onTour: () => void }) {
  const [show, setShow] = useState(false);
  useEffect(() => {
    let seen = false;
    try {
      seen = window.localStorage.getItem(KEY) === "1";
    } catch {
      // storage blocked: show it, it can still be closed for this visit
    }
    setShow(!seen);
  }, []);
  if (!show) return null;
  const close = () => {
    setShow(false);
    try {
      window.localStorage.setItem(KEY, "1");
    } catch {
      // nothing to remember
    }
  };
  return (
    <section
      aria-labelledby="intro-h"
      data-testid="intro-card"
      className={`theme-paper cw-pop-in rounded-xl bg-surface p-4 text-ink shadow-[0_1px_2px_rgba(6,17,30,0.14),0_6px_18px_rgba(6,17,30,0.24)] ${className}`}
    >
      <h2 id="intro-h" className="font-display text-[19px] font-medium leading-snug">
        Harmful algal blooms on the California coast
      </h2>
      <p className="mt-1 text-[14px] leading-snug text-ink-2">
        NOAA&apos;s bloom forecast, satellite and shore-station measurements, and official notices, each labelled for what it is. CoastWatch is not an official source: closures and advisories come from CDFW and CDPH.
      </p>
      <p className="mt-2 flex flex-wrap gap-1.5" aria-label="Kinds of information">
        <KindChip kind="model" />
        <KindChip kind="measured" />
        <KindChip kind="official" />
      </p>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <Link href="/about" className="text-[14px] font-medium text-accent underline-offset-2 hover:underline" data-testid="intro-about">
          About CoastWatch
        </Link>
        <div className="flex gap-2">
          <button type="button" onClick={close} className="min-h-[40px] rounded-md px-3 text-[14px] font-medium text-ink-2 hover:text-ink" data-testid="intro-dismiss">
            Not now
          </button>
          <button
            type="button"
            onClick={() => {
              setShow(false);
              onTour();
            }}
            className="min-h-[40px] rounded-md bg-navy-900 px-3.5 text-[14px] font-medium text-white hover:bg-navy-800"
            data-testid="intro-tour"
          >
            Take the tour
          </button>
        </div>
      </div>
    </section>
  );
}
