import type { ReactNode } from "react";
import s from "./catalog.module.css";

/** One component: its name, its job, and its specimens. */
export function Section({ id, name, purpose, children }: { id: string; name: string; purpose: ReactNode; children: ReactNode }) {
  return (
    <section id={id} className={s.section} aria-labelledby={`${id}-h`}>
      <header className={s.sectionHead}>
        <h2 id={`${id}-h`} className={s.sectionName}>
          {name}
        </h2>
        <p className={s.purpose}>{purpose}</p>
      </header>
      <div className={s.specimens}>{children}</div>
    </section>
  );
}

/** One state of a component, captioned. */
export function Specimen({ state, children, wide, note }: { state: string; children: ReactNode; wide?: boolean; note?: ReactNode }) {
  return (
    <div className={wide ? `${s.specimen} ${s.wide}` : s.specimen}>
      <p className={s.state}>
        {state}
        {note ? <span className={s.stateNote}> · {note}</span> : null}
      </p>
      <div className={s.specimenBody}>{children}</div>
    </div>
  );
}
