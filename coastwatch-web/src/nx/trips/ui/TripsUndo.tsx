"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useState } from "react";
import { Button, IconButton, register, registerFrom, registerTo } from "@/nx/ds";
import { runPendingUndo, setPendingUndo, usePendingUndo } from "../undo";
import { notifyTrips } from "../useTripsData";
import s from "./economics.module.css";

/**
 * "Deleted "…". Undo": the trip deleted last in this tab, put back exactly as it was. Reads the tab's pending undo
 * (../undo.ts), so it stays visible when deleting the last trip turns the history into the empty state: the history
 * renders it, and the empty state can place it too.
 */
export function TripsUndo({ onChanged = notifyTrips }: { onChanged?: () => void }) {
  const undo = usePendingUndo();
  const reduce = useReducedMotion();
  const [failed, setFailed] = useState<string | null>(null);
  const t = reduce ? { duration: 0 } : register;
  return (
    <>
      <AnimatePresence>
        {undo ? (
          <motion.div className={s.hlUndo} role="status" initial={registerFrom} animate={registerTo} exit={{ opacity: 0 }} transition={t} data-testid="history-undo">
            <span>Deleted “{undo.name}”.</span>
            <Button
              variant="quiet"
              onClick={() => {
                const r = runPendingUndo();
                setFailed(r && !r.ok ? (r.reason === "quota" ? "This browser's storage is full: the trip could not be put back." : "The trip could not be put back.") : null);
                onChanged();
              }}
            >
              Undo
            </Button>
            <IconButton icon="close" size={44} label="Dismiss" onClick={() => setPendingUndo(null)} />
          </motion.div>
        ) : null}
      </AnimatePresence>
      {failed ? (
        <p className={s.hlNotice} role="alert">
          {failed}
        </p>
      ) : null}
    </>
  );
}
