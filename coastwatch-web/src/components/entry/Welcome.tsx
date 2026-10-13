"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { PlacePicker } from "@/components/coast/PlacePicker";
import { coastHref, type Audience, type CoastPlace } from "@/lib/coast";
import { defaultAudience, usePurpose, type PurposeChoice } from "@/lib/purpose";
import { writeMode } from "@/lib/mode";
import { TOUR_SEEN_KEY } from "@/components/map/Tour";

/**
 * The front door (first visit to the map, never over a shared link): "Explore California's coast"
 * and three obvious paths. It sets defaults only (purpose, presentation, where to start), never a
 * fact. Returning visitors get one line back to their waters instead.
 */
export const WATERS_KEY = "cw-my-waters";
const HOME_KEY = "cw-home-port";

const read = (k: string) => {
  try {
    return window.localStorage.getItem(k);
  } catch {
    return null;
  }
};
const write = (k: string, v: string) => {
  try {
    window.localStorage.setItem(k, v);
  } catch {
    // not remembered on this device
  }
};

type Path = "fish" | "conditions" | "research";
const PATHS: { id: Path; title: string; text: string; art: React.ReactNode }[] = [
  {
    id: "fish",
    title: "Plan a fishing trip",
    text: "Your waters, your species: what officially applies, the sea, and what to check before you go.",
    art: (
      <svg viewBox="0 0 64 40" className="h-10 w-16" aria-hidden>
        <path d="M4 30c8-6 16-6 24 0s16 6 24 0" fill="none" stroke="#74cfe6" strokeWidth="2.2" strokeLinecap="round" />
        <path d="M20 16c6-6 14-6 20 0-6 6-14 6-20 0Zm20 0 6-4v8z" fill="#f2b04c" />
      </svg>
    ),
  },
  {
    id: "conditions",
    title: "Explore coastal conditions",
    text: "The seafloor, the forecast, satellite views and currents, on a map of the whole coast.",
    art: (
      <svg viewBox="0 0 64 40" className="h-10 w-16" aria-hidden>
        <path d="M6 32c10-14 20-22 30-22 8 0 14 6 22 4" fill="none" stroke="#74cfe6" strokeWidth="2.2" strokeLinecap="round" />
        <circle cx="36" cy="10" r="4" fill="#4fd1ae" />
        <path d="M6 22h52" stroke="#f2b04c" strokeWidth="2" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: "research",
    title: "Analyze scientific data",
    text: "Shore-station records, model values and satellite coverage, with every source and date.",
    art: (
      <svg viewBox="0 0 64 40" className="h-10 w-16" aria-hidden>
        <path d="M6 34 18 22l10 6 12-16 16 8" fill="none" stroke="#b58cf0" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
        {[18, 28, 40].map((x, i) => (
          <circle key={x} cx={x} cy={[22, 28, 12][i]} r="2.6" fill="#4fd1ae" />
        ))}
      </svg>
    ),
  },
];

const ease = [0.2, 0, 0, 1] as const;

export function Welcome({ places, suppressed, onConditions, pillClass = "", onOpenChange }: { places: CoastPlace[]; suppressed: boolean; onConditions: () => void; pillClass?: string; onOpenChange?: (open: boolean) => void }) {
  const router = useRouter();
  const [purpose, setPurpose, ready] = usePurpose();
  const [show, setShow] = useState<"welcome" | "return" | null>(null);
  const [step, setStep] = useState<"paths" | "fish">("paths");
  const [kind, setKind] = useState<Audience>("commercial");
  const [resume, setResume] = useState<CoastPlace | null>(null);

  useEffect(() => {
    if (!ready || suppressed) return;
    if (read(TOUR_SEEN_KEY) !== "1") return setShow("welcome");
    const k = read(HOME_KEY) ?? read(WATERS_KEY);
    const h = k ? (places.find((p) => p.key === k) ?? null) : null;
    if (h && purpose && purpose.purpose !== "research" && purpose.purpose !== "exploring") {
      setResume(h);
      setShow("return");
    }
  }, [ready, suppressed, purpose, places]);

  useEffect(() => onOpenChange?.(show === "welcome"), [show, onOpenChange]);
  const close = () => {
    write(TOUR_SEEN_KEY, "1");
    setShow(null);
  };
  const choose = (p: Path) => {
    if (p === "fish") return setStep("fish");
    if (p === "conditions") {
      setPurpose({ purpose: "exploring" });
      close();
      onConditions();
      return;
    }
    setPurpose({ purpose: "research" });
    writeMode("research");
    close();
    router.push("/bloom");
  };
  const toCoast = (place: CoastPlace) => {
    const c: PurposeChoice = kind === "commercial" ? { purpose: "commercial" } : { purpose: "recreational", harvest: kind === "shellfish" ? "shellfish" : "fishing" };
    setPurpose(c);
    write(WATERS_KEY, place.key);
    close();
    router.push(coastHref({ place: place.key, for: defaultAudience(c) }));
  };

  return (
    <AnimatePresence>
      {show === "return" && resume && purpose && (
        <motion.div
          key="return"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.32, ease, delay: 0.4 }}
          className={`pointer-events-auto absolute z-20 ${pillClass}`}
        >
          <Link
            href={coastHref({ place: resume.key, for: defaultAudience(purpose) })}
            className="cw-press theme-paper flex min-h-[44px] w-fit max-w-full items-center gap-2.5 rounded-full bg-surface py-1 pl-1.5 pr-4 text-[14px] text-ink shadow-[0_8px_28px_rgba(6,17,30,0.35)] hover:bg-surface-2"
            data-testid="purpose-return"
          >
            <span className="grid h-8 w-8 place-items-center rounded-full bg-navy-900 text-[13px] text-white" aria-hidden>
              ↺
            </span>
            <span className="truncate">
              Continue with <b className="font-semibold">{resume.name}</b>
            </span>
            <span aria-hidden className="text-ink-3">→</span>
          </Link>
        </motion.div>
      )}
      {show === "welcome" && (
        <motion.section
          key="welcome"
          aria-labelledby="welcome-h"
          data-testid="intro-card"
          data-step={step === "paths" ? 1 : 2}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, transition: { duration: 0.35 } }}
          transition={{ duration: 0.5 }}
          className="pointer-events-auto absolute inset-0 z-40 flex flex-col overflow-y-auto bg-[linear-gradient(180deg,rgba(6,17,30,0.9)_0%,rgba(6,17,30,0.8)_45%,rgba(6,17,30,0.96)_100%)] px-4 pb-6 pt-6 text-white sm:px-10 sm:pt-10 lg:justify-center lg:bg-[linear-gradient(90deg,rgba(6,17,30,0.96)_0%,rgba(6,17,30,0.86)_42%,rgba(6,17,30,0.15)_78%)] lg:px-16 lg:pb-10"
        >
          <div className="w-full max-w-[640px]">
            <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease, delay: 0.15 }} className="text-[14px] font-medium text-[#74cfe6]">
              CoastWatch · harmful algal blooms, California
            </motion.p>
            <motion.h1
              id="welcome-h"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease, delay: 0.25 }}
              className="mt-2 font-display text-[40px] font-semibold leading-[1.02] tracking-[-0.01em] sm:text-[58px]"
            >
              Explore California&apos;s coast
            </motion.h1>
            <motion.p initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.55, ease, delay: 0.38 }} className="mt-3 max-w-[52ch] text-[16px] leading-relaxed text-[#c6d3e1] sm:text-[17px]">
              Official notices, what has been measured, and what NOAA&apos;s model expects, for the waters you care about. CoastWatch is not an official source: closures come from CDFW and CDPH.
            </motion.p>

            <AnimatePresence mode="wait" initial={false}>
              {step === "paths" ? (
                <motion.ul key="paths" className="mt-7 grid gap-2.5 sm:gap-3 lg:grid-cols-3" initial="hidden" animate="show" exit={{ opacity: 0, x: -20, transition: { duration: 0.2 } }} variants={{ show: { transition: { staggerChildren: 0.08, delayChildren: 0.5 } } }}>
                  {PATHS.map((p) => (
                    <motion.li key={p.id} variants={{ hidden: { opacity: 0, y: 18 }, show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 260, damping: 26 } } }}>
                      <motion.button
                        type="button"
                        onClick={() => choose(p.id)}
                        data-testid={`welcome-${p.id}`}
                        whileHover={{ y: -3 }}
                        whileTap={{ scale: 0.98 }}
                        className="group flex h-full w-full items-center gap-4 rounded-2xl border border-white/12 bg-white/[0.06] p-4 text-left backdrop-blur-md transition-colors hover:border-white/30 hover:bg-white/[0.11] lg:flex-col lg:items-start lg:gap-3 lg:p-5"
                      >
                        <span className="shrink-0">{p.art}</span>
                        <span className="min-w-0">
                          <span className="block text-[17px] font-semibold leading-snug">{p.title}</span>
                          <span className="mt-1 block text-[14px] leading-snug text-[#aebdcd]">{p.text}</span>
                        </span>
                        <span aria-hidden className="ml-auto text-[20px] text-[#74cfe6] transition-transform group-hover:translate-x-1 lg:hidden">
                          →
                        </span>
                      </motion.button>
                    </motion.li>
                  ))}
                </motion.ul>
              ) : (
                <motion.div key="fish" className="theme-paper mt-7 rounded-2xl bg-surface p-4 text-ink shadow-[0_18px_50px_rgba(0,0,0,0.35)] sm:p-5" initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 24 }} transition={{ type: "spring", stiffness: 260, damping: 28 }}>
                  <p className="text-[13px] font-medium text-accent">Plan a fishing trip · 1 of 2</p>
                  <h2 className="mt-1 font-display text-[24px] font-medium leading-tight">Where will you fish?</h2>
                  <div role="radiogroup" aria-label="Kind of fishing" className="mt-3 flex flex-wrap gap-1.5">
                    {(
                      [
                        ["commercial", "Commercial"],
                        ["recreational", "Recreational"],
                        ["shellfish", "Shellfish harvest"],
                      ] as const
                    ).map(([id, label]) => (
                      <button
                        key={id}
                        type="button"
                        role="radio"
                        aria-checked={kind === id}
                        onClick={() => setKind(id)}
                        data-testid={`intro-purpose-${id}`}
                        className={`cw-press min-h-[40px] rounded-full border px-3.5 text-[14px] ${kind === id ? "border-navy-900 bg-navy-900 text-white" : "border-hairline-strong text-ink-2 hover:border-ink-3"}`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  <div className="mt-3">
                    <PlacePicker places={places} current={null} onPick={toCoast} />
                  </div>
                  <div className="mt-3 flex items-center justify-between text-[14px]">
                    <button type="button" onClick={() => setStep("paths")} className="min-h-[36px] text-ink-3 hover:text-ink" data-testid="intro-back">
                      ← Back
                    </button>
                    <span className="text-ink-3">Search a port, beach or harvest site</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9 }} className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-[14px] text-[#aebdcd]">
              <button type="button" onClick={close} className="min-h-[36px] font-medium text-white underline decoration-white/40 underline-offset-4 hover:decoration-white" data-testid="intro-dismiss">
                Skip to the map
              </button>
              <Link href="/about" className="min-h-[36px] leading-[36px] hover:text-white" data-testid="intro-about">
                About CoastWatch
              </Link>
            </motion.div>
          </div>
        </motion.section>
      )}
    </AnimatePresence>
  );
}
