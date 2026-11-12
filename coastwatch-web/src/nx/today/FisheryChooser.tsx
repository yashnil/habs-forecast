"use client";

import { motion } from "motion/react";
import type { FisheryId } from "@/pilot/contracts";
import { FISHERY_OPTIONS } from "@/pilot/facts";
import { pan, press } from "@/nx/ds/motion";
import { Skeleton } from "@/nx/ds/DataState";
import { PORT_NAME_LAYOUT } from "./layout";
import { suggestFisheries, type TodayInputs } from "./model";
import s from "./today.module.css";


/**
 * Which fishery Today leads with (journeys.md Journey 1). Nothing is chosen for the reader. The fisheries
 * Neritic has a test site or a place-specific record for at this port come first, labelled as a suggestion
 * (port-level landings are not published openly; src/content/landings.ts holds port areas only). One tap
 * chooses and Today assembles in place.
 */
export function FisheryChooser({ port, inputs, onChoose }: { port: { key: string; name: string }; inputs: TodayInputs | null | undefined; onChoose: (f: FisheryId) => void }) {
  const suggested = inputs ? suggestFisheries(inputs) : [];
  const rest = FISHERY_OPTIONS.filter((f) => !suggested.includes(f.id));
  const option = (id: FisheryId, label: string) => (
    <li key={id}>
      {/* design-lint-allow raw-button: a full-width list row that presses with Motion (`press`); ds OptionList is a radio group, and choosing here is an action that leaves the list */}
      <motion.button type="button" className={s.option} onClick={() => onChoose(id)} data-fishery={id} {...press}>
        {label}
      </motion.button>
    </li>
  );
  return (
    <div className={s.chooser} data-testid="nx-fishery-chooser">
      <motion.h1 id="nx-today-title" layoutId={PORT_NAME_LAYOUT} className={s.chooserTitle} transition={pan}>
        {port.name}
      </motion.h1>
      <p className={s.chooserLead}>What do you fish from here? Today leads with it, and you can change it at any time under Others.</p>
      {inputs === undefined ? (
        // the list's own geometry while the port's records are read: a head and its rows
        <Skeleton className={s.block} label={`Reading what Neritic has for ${port.name}`} lines={6} />
      ) : suggested.length ? (
        <>
          <h2 className={s.groupHead}>Suggested: Neritic has records for these here</h2>
          <ul className={s.options}>{suggested.map((id) => option(id, FISHERY_OPTIONS.find((f) => f.id === id)!.label))}</ul>
        </>
      ) : null}
      {inputs !== undefined ? (
        <>
          <h2 className={s.groupHead}>{suggested.length ? "Other fisheries" : "Fisheries"}</h2>
          <ul className={s.options}>{rest.map((f) => option(f.id, f.label))}</ul>
        </>
      ) : null}
    </div>
  );
}
