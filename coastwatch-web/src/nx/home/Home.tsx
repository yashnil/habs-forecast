"use client";

import Link from "next/link";
import { useLayoutEffect, useState } from "react";
import { Button } from "@/nx/ds/Button";
import { PortFinder, type PortEntry } from "@/nx/ds/PortFinder";
import { Statement } from "@/nx/ds/Statement";
import { SkipOpening } from "@/nx/opening/Opening";
import { PORT_NAME_LAYOUT } from "@/nx/today/layout";
import type { FinderData } from "./finder.server";
import s from "./home.module.css";

export type Picked = { key: string; name: string };

/**
 * Home, a first visit (journeys.md Journey 1): what Neritic is in one sentence, the one action (Find your
 * port), and three quiet paths. No statewide figures: nothing is said about the water until there is a port.
 * Find your port opens the finder in place (ds PortFinder: type, or read down the coast by CDFW port area).
 */
export function Home({ finder, onPick }: { finder: FinderData; onPick: (p: Picked) => void }) {
  const [step, setStep] = useState<"intro" | "find">("intro");
  // the chosen name first takes the shared layoutId where it stands in the list; once that is drawn, the
  // port is handed on and the name travels into the chooser's title (README §8 `layout`)
  const [chosen, setChosen] = useState<PortEntry | null>(null);
  useLayoutEffect(() => {
    if (!chosen) return;
    const key = finder.keys[chosen.code];
    if (key) onPick({ key, name: chosen.name });
  }, [chosen, finder.keys, onPick]);
  return (
    <section className={s.home} data-nx-home data-testid="nx-home" data-step={step}>
      {step === "intro" ? (
        <>
          <Statement className={s.lead} size="md" as="h1" scope="Free, for California commercial fishermen and coastal researchers" title="Season status, crab tests, whale measures and the next decision date for your port." />
          <div className={s.actions}>
            <Button variant="primary" onClick={() => setStep("find")} data-testid="nx-find-port">
              Find your port
            </Button>
            <SkipOpening />
          </div>
          <nav className={s.paths} aria-label="Other ways in">
            <Link href="/explore">See the water</Link>
            <Link href="/coast">Plan a trip</Link>
            <Link href="/research">Check records</Link>
          </nav>
        </>
      ) : (
        <div className={s.find} data-testid="nx-home-finder">
          <div className={s.findHead}>
            <h1 className={s.findTitle}>Find your port</h1>
            <button type="button" className={s.back} onClick={() => setStep("intro")}>
              Back
            </button>
          </div>
          {finder.ports.length ? (
            <PortFinder
              className={s.finder}
              ports={finder.ports}
              regions={finder.regions}
              onSelect={(p) => finder.keys[p.code] && setChosen(p)}
              nameLayoutId={(p) => (p.code === chosen?.code ? PORT_NAME_LAYOUT : undefined)}
              autoFocus
              label="Find your port"
            />
          ) : (
            <p className={s.empty}>The port list could not be read just now. Try again in a minute, or open a page from the bar.</p>
          )}
        </div>
      )}
    </section>
  );
}
