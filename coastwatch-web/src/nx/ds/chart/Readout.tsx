"use client";
import { motion } from "motion/react";
import type { ReactNode } from "react";
import { latLon } from "../format";
import { Icon } from "../Icon";
import { Mark } from "../Mark";
import { registerFrom, registerTo, sheet } from "../motion";
import { Stamp, type Provenance } from "../Stamp";
import { cx } from "../util";
import s from "./readout.module.css";

export type ReadoutState = "value" | "loading" | "noview" | "error";

export interface ReadoutProps {
  /** What was read: "Chlorophyll at this point". */
  title: string;
  lat: number;
  lon: number;
  state: ReadoutState;
  value?: string;
  unit?: string;
  provenance?: Provenance;
  now?: string;
  /** Mandatory words and pixel age: "Algae, not toxin. This pixel was seen Nov 8, 2 days ago." */
  note?: ReactNode;
  /** The linked transect or nearby record (a SeriesChart). */
  children?: ReactNode;
  /** The next step: "See the record nearby". */
  action?: ReactNode;
  /** Loading / no view / error wording; each says what is missing and what to do. */
  message?: ReactNode;
  onClose?: () => void;
  className?: string;
}

/** The result of tapping the water: value, kind, date, pixel age and a linked chart (spec section 9). */
export function Readout({ title, lat, lon, state, value, unit, provenance, now, note, children, action, message, onClose, className }: ReadoutProps) {
  return (
    <motion.section className={cx(s.readout, className)} aria-label={title} aria-live="polite" initial={registerFrom} animate={registerTo} transition={sheet} data-state={state}>
      <header className={s.head}>
        <div>
          <h3 className={s.title}>{title}</h3>
          <p className={s.coord}>{latLon(lat, lon)}</p>
        </div>
        {onClose ? (
          <button type="button" className={s.close} onClick={onClose} aria-label="Close the reading">
            <Icon name="close" />
          </button>
        ) : null}
      </header>
      {state === "value" ? (
        <>
          <p className={s.big}>
            {value}
            {unit ? <small className={s.unit}>{unit}</small> : null}
          </p>
          {provenance ? (
            <p className={s.stamp}>
              <Stamp provenance={provenance} now={now} />
            </p>
          ) : null}
          {note ? <p className={s.note}>{note}</p> : null}
          {children ? <div className={s.chart}>{children}</div> : null}
          {action ? <div className={s.action}>{action}</div> : null}
        </>
      ) : state === "loading" ? (
        <p className={s.message} role="status">
          {message ?? "Reading this point…"}
        </p>
      ) : state === "noview" ? (
        <div className={s.noview}>
          <Mark kind="noview" />
          <p className={s.message}>{message ?? "No view here. No value is not a low value."}</p>
        </div>
      ) : (
        <p className={s.message}>{message ?? "This point couldn't be read. Try again, or open the layer's source."}</p>
      )}
    </motion.section>
  );
}
