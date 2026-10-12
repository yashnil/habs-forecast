"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { PlacePicker } from "@/components/coast/PlacePicker";
import { Icon } from "@/components/ui/Icon";
import { coastHref, type CoastPlace } from "@/lib/coast";
import { defaultAudience, PURPOSES, usePurpose, type Purpose, type PurposeChoice } from "@/lib/purpose";
import { TOUR_SEEN_KEY as KEY } from "./Tour";

const HOME_KEY = "cw-home-port";
export const WATERS_KEY = "cw-my-waters";

function seen(): boolean {
  try {
    return window.localStorage.getItem(KEY) === "1";
  } catch {
    return false; // storage blocked: offer it; it can still be closed for this visit
  }
}
function markSeen() {
  try {
    window.localStorage.setItem(KEY, "1");
  } catch {
    // nothing to remember
  }
}
function homePort(): string | null {
  try {
    return window.localStorage.getItem(HOME_KEY);
  } catch {
    return null;
  }
}

/**
 * The first visit's welcome on the map (the single introduction; never a modal, never in front of
 * official information): what CoastWatch is, then an optional "what brings you here?". The choice
 * only sets defaults (My Coast's filter and section order) and where the visitor starts; "Just
 * exploring" starts the guided tour. Skip, the X and Back are always there. A shared link (any
 * parameter in the address) takes precedence: the welcome waits for a plain visit.
 *
 * After the first visit, someone who chose fishing or ports and saved a home port sees one small
 * link back to their waters instead.
 */
export function IntroCard({ className = "", onTour, places, suppressed = false }: { className?: string; onTour: () => void; places: CoastPlace[]; suppressed?: boolean }) {
  const router = useRouter();
  const [show, setShow] = useState<"welcome" | "return" | null>(null);
  const [step, setStep] = useState<1 | 2>(1);
  const [choice, setChoice] = useState<PurposeChoice | null>(null);
  const [place, setPlace] = useState<CoastPlace | null>(null);
  const [saveHome, setSaveHome] = useState(false);
  const [purpose, setPurpose, ready] = usePurpose();
  const [home, setHome] = useState<CoastPlace | null>(null);

  useEffect(() => {
    if (!ready || suppressed) return;
    if (!seen()) return setShow("welcome");
    let hk = homePort();
    try {
      hk = hk ?? window.localStorage.getItem(WATERS_KEY);
    } catch {
      // nothing stored
    }
    const h = hk ? (places.find((p) => p.key === hk) ?? null) : null;
    if (h && purpose && purpose.purpose !== "research" && purpose.purpose !== "exploring") {
      setHome(h);
      setShow("return");
    }
  }, [ready, suppressed, purpose, places]);

  if (!show) return null;
  const done = () => {
    markSeen();
    setShow(null);
  };

  if (show === "return" && home && purpose) {
    return (
      <Link
        href={coastHref({ place: home.key, for: defaultAudience(purpose) })}
        className={`theme-paper flex min-h-[44px] w-fit max-w-full items-center gap-2 rounded-full bg-surface px-3.5 text-[14px] font-medium text-ink shadow-[0_6px_20px_rgba(6,17,30,0.3)] hover:bg-surface-2 ${className}`}
        data-testid="purpose-return"
      >
        <Icon name="pin" className="h-4 w-4 text-accent" />
        <span className="truncate">Your waters: {home.name}</span>
        <span aria-hidden className="text-ink-3">→</span>
      </Link>
    );
  }

  const pick = (id: Purpose) => {
    const c: PurposeChoice = id === "recreational" ? { purpose: id, harvest: "fishing" } : { purpose: id };
    setChoice(c);
    if (id === "exploring") {
      setPurpose(c);
      done();
      onTour();
      return;
    }
    setStep(2);
  };
  // the purpose is saved when the visitor finishes step 2, not when they only look at it
  const finish = () => {
    if (choice) setPurpose(choice);
    done();
  };
  const toCoast = () => {
    if (place) {
      try {
        window.localStorage.setItem(WATERS_KEY, place.key);
      } catch {
        // not remembered
      }
    }
    if (place && place.kind === "port" && (saveHome || choice?.purpose === "ports")) {
      try {
        window.localStorage.setItem(HOME_KEY, place.key);
      } catch {
        // not remembered
      }
    }
    finish();
    router.push(coastHref({ place: place?.key ?? null, for: defaultAudience(choice) }));
  };

  const fishing = choice?.purpose === "commercial" || choice?.purpose === "recreational" || choice?.purpose === "ports";
  const usable = choice?.purpose === "ports" ? places.filter((p) => p.kind === "port") : places;

  return (
    <section
      aria-labelledby="intro-h"
      data-testid="intro-card"
      data-step={step}
      className={`theme-paper cw-pop-in rounded-xl bg-surface p-4 text-ink shadow-[0_1px_2px_rgba(6,17,30,0.14),0_6px_18px_rgba(6,17,30,0.24)] ${className}`}
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-ink-3" data-testid="intro-progress">
          Welcome · {step} of 2
        </p>
        <button type="button" onClick={done} aria-label="Close the welcome" className="-mr-2 -mt-2 grid h-10 w-10 shrink-0 place-items-center rounded-md text-ink-3 hover:bg-surface-3 hover:text-ink" data-testid="intro-close">
          <Icon name="close" className="h-5 w-5" />
        </button>
      </div>

      {step === 1 ? (
        <>
          <h2 id="intro-h" className="mt-1 font-display text-[20px] font-medium leading-snug">
            Harmful algal blooms on the California coast
          </h2>
          <p className="mt-1 text-[14px] leading-snug text-ink-2">
            NOAA&apos;s forecast model, measured shore-station and satellite data, and official notices, each labelled for what it is. CoastWatch is not an official source.
          </p>
          <h3 className="mt-3 text-[14px] font-semibold text-ink" id="intro-q">
            What brings you here? <span className="font-normal text-ink-3">(optional)</span>
          </h3>
          <ul className="mt-2 space-y-1 sm:space-y-1.5" aria-labelledby="intro-q">
            {PURPOSES.map((p) => (
              <li key={p.id}>
                <button
                  type="button"
                  onClick={() => pick(p.id)}
                  className="flex min-h-[44px] w-full items-center justify-between gap-3 rounded-lg border border-hairline-strong px-3 py-1.5 text-left hover:border-accent hover:bg-surface-2"
                  data-testid={p.id === "exploring" ? "intro-tour" : `intro-purpose-${p.id}`}
                  aria-describedby={`intro-hint-${p.id}`}
                >
                  <span>
                    <span className="block text-[15px] font-medium leading-snug text-ink">{p.label}</span>
                    <span id={`intro-hint-${p.id}`} className={p.id === "exploring" ? "block text-[13px] text-ink-3 max-[389px]:sr-only" : "sr-only"}>
                      {p.hint}
                    </span>
                  </span>
                  <span aria-hidden className="text-ink-3">→</span>
                </button>
              </li>
            ))}
          </ul>
          <div className="mt-3 flex items-center justify-between gap-3">
            <Link href="/about" className="text-[14px] font-medium text-accent underline-offset-2 hover:underline" data-testid="intro-about">
              About CoastWatch
            </Link>
            <button type="button" onClick={done} className="min-h-[40px] rounded-md px-3 text-[14px] font-medium text-ink-2 hover:text-ink" data-testid="intro-dismiss">
              Skip
            </button>
          </div>
        </>
      ) : (
        <>
          {fishing ? (
            <>
              <h2 id="intro-h" className="mt-1 font-display text-[20px] font-medium leading-snug">
                {choice?.purpose === "ports" ? "Which port?" : "Which waters do you care about?"}
              </h2>
              <p className="mt-1 text-[14px] text-ink-2">My Coast shows the official notices, nearby measurements and forecast availability for them. You can change this any time.</p>
              {choice?.purpose === "recreational" && (
                <div role="group" aria-label="Fishing or shellfish" className="mt-3 flex gap-1.5">
                  {(["fishing", "shellfish"] as const).map((h) => (
                    <button
                      key={h}
                      type="button"
                      aria-pressed={choice.harvest === h}
                      onClick={() => setChoice({ purpose: "recreational", harvest: h })}
                      className={`min-h-[40px] rounded-full border px-3.5 text-[14px] ${choice.harvest === h ? "border-navy-900 bg-navy-900 text-white" : "border-hairline-strong text-ink-2"}`}
                      data-testid={`intro-harvest-${h}`}
                    >
                      {h === "fishing" ? "Fishing" : "Shellfish"}
                    </button>
                  ))}
                </div>
              )}
              <div className="mt-3">
                <PlacePicker places={usable} current={place} onPick={setPlace} label="Choose waters" />
              </div>
              {place && (
                <p className="mt-2 text-[14px] text-ink" data-testid="intro-place">
                  {place.name}
                  {place.kind === "port" && (
                    <label className="ml-1 inline-flex items-center gap-1.5 text-ink-2">
                      · <input type="checkbox" checked={saveHome || choice?.purpose === "ports"} disabled={choice?.purpose === "ports"} onChange={(e) => setSaveHome(e.target.checked)} data-testid="intro-save-home" /> my home port
                    </label>
                  )}
                </p>
              )}
              <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
                <button type="button" onClick={() => setStep(1)} className="min-h-[40px] rounded-md border border-hairline-strong px-3 text-[14px] font-medium text-ink" data-testid="intro-back">
                  Back
                </button>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      finish();
                      onTour();
                    }}
                    className="min-h-[40px] px-2 text-[14px] text-ink-2 hover:text-ink"
                    data-testid="intro-step2-tour"
                  >
                    Map tour
                  </button>
                  <button
                    type="button"
                    onClick={toCoast}
                    className="min-h-[40px] rounded-md bg-navy-900 px-3.5 text-[14px] font-medium text-white hover:bg-navy-800"
                    data-testid="intro-to-coast"
                  >
                    {place ? `Show ${place.name}` : "Go to My Coast"}
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              <h2 id="intro-h" className="mt-1 font-display text-[20px] font-medium leading-snug">
                Where would you like to start?
              </h2>
              <ul className="mt-3 space-y-1.5">
                {[
                  { href: "/bloom", label: "Shore-station measurements", hint: "Domoic acid and Pseudo-nitzschia at 17 piers, with history (Bloom Intelligence)", id: "bloom" },
                  { href: "/coast", label: "Measurements and model coverage for chosen waters", hint: "My Coast, measurements and forecast coverage first", id: "coast" },
                ].map((x) => (
                  <li key={x.id}>
                    <Link
                      href={x.href}
                      onClick={finish}
                      className="flex min-h-[44px] items-center justify-between gap-3 rounded-lg border border-hairline-strong px-3 py-2 hover:border-accent hover:bg-surface-2"
                      data-testid={`intro-research-${x.id}`}
                    >
                      <span>
                        <span className="block text-[15px] font-medium text-ink">{x.label}</span>
                        <span className="block text-[13px] text-ink-3">{x.hint}</span>
                      </span>
                      <span aria-hidden className="text-ink-3">→</span>
                    </Link>
                  </li>
                ))}
                <li>
                  <button
                    type="button"
                    onClick={() => {
                      finish();
                      onTour();
                    }}
                    className="flex min-h-[44px] w-full items-center justify-between gap-3 rounded-lg border border-hairline-strong px-3 py-2 text-left hover:border-accent hover:bg-surface-2"
                    data-testid="intro-research-tour"
                  >
                    <span>
                      <span className="block text-[15px] font-medium text-ink">The forecast and satellite map</span>
                      <span className="block text-[13px] text-ink-3">A short tour of C-HARM, satellite chlorophyll and the gaps</span>
                    </span>
                    <span aria-hidden className="text-ink-3">→</span>
                  </button>
                </li>
              </ul>
              <div className="mt-3 flex items-center justify-between">
                <button type="button" onClick={() => setStep(1)} className="min-h-[40px] rounded-md border border-hairline-strong px-3 text-[14px] font-medium text-ink" data-testid="intro-back">
                  Back
                </button>
                <button type="button" onClick={finish} className="min-h-[40px] rounded-md px-3 text-[14px] font-medium text-ink-2 hover:text-ink" data-testid="intro-dismiss">
                  Skip
                </button>
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
