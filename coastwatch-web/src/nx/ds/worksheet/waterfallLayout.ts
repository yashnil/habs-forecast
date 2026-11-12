/**
 * Waterfall layout, DOM-free. Fixes E3: the domain's low end is the running minimum, so when costs exceed
 * gross the bars stay inside the chart instead of running below the baseline.
 */
export type StepKind = "gross" | "cost" | "net";

export interface WaterfallStep {
  id: string;
  label: string;
  /** Gross and net as signed totals; a cost as a positive amount taken off. Null: not worked out yet. */
  amount: number | null;
  kind: StepKind;
  /** Words when the amount is null: "waiting for price". Never 0. */
  waiting?: string;
}

export interface Bar {
  id: string;
  kind: StepKind;
  /** Bar ends in money; from < to. */
  from: number;
  to: number;
  /** Where the previous bar ended (the connector's position). */
  joinAt: number | null;
  waiting: boolean;
}

export interface WaterfallLayout {
  bars: Bar[];
  domain: [number, number];
  /** Position 0 to 1 of a money value inside the domain. */
  pos: (v: number) => number;
}

export function layoutWaterfall(steps: WaterfallStep[]): WaterfallLayout {
  let running = 0;
  let lo = 0;
  let hi = 0;
  let prevEnd: number | null = null;
  let broken = false;
  const bars: Bar[] = steps.map((st) => {
    if (st.amount === null || broken) {
      if (st.kind !== "net") broken = broken || st.amount === null;
      return { id: st.id, kind: st.kind, from: 0, to: 0, joinAt: null, waiting: true };
    }
    let from: number, to: number, joinAt: number | null;
    if (st.kind === "gross") {
      running = st.amount;
      from = Math.min(0, running);
      to = Math.max(0, running);
      joinAt = null;
    } else if (st.kind === "cost") {
      const start = running;
      running = running - st.amount;
      from = Math.min(start, running);
      to = Math.max(start, running);
      joinAt = prevEnd;
    } else {
      from = Math.min(0, st.amount);
      to = Math.max(0, st.amount);
      joinAt = prevEnd;
      running = st.amount;
    }
    lo = Math.min(lo, from, running);
    hi = Math.max(hi, to, running);
    prevEnd = running;
    return { id: st.id, kind: st.kind, from, to, joinAt, waiting: false };
  });
  if (hi === lo) hi = lo + 1;
  const span = hi - lo;
  return { bars, domain: [lo, hi], pos: (v: number) => (v - lo) / span };
}
