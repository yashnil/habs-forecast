"use client";

/**
 * What the map shows and when: one line per layer drawn (observation, model, or a stated
 * gap), with its own observation or valid time. It heads the layer dock, so the time is
 * visible whenever the map is, on every screen size.
 */
export type StampLine = { kind: "observation" | "model" | "gap"; text: string; testid: string };

const DOT: Record<StampLine["kind"], string> = {
  model: "rounded-full bg-[var(--cw-forecast)]",
  observation: "rounded-full bg-[var(--cw-chl)]",
  gap: "rounded-[2px] border border-dashed border-warning bg-transparent",
};

export function MapStamp({ lines, className = "" }: { lines: StampLine[]; className?: string }) {
  if (!lines.length) return null;
  return (
    <div data-testid="map-stamp" className={`flex flex-col gap-0.5 ${className}`}>
      {lines.map((l, i) => (
        <p key={l.testid} data-testid={l.testid} className={`flex items-baseline gap-1.5 leading-snug ${i === 0 ? "text-[13px] font-semibold text-ink" : l.kind === "gap" ? "text-[12px] text-warning" : "text-[12.5px] font-medium text-ink"}`}>
          <span aria-hidden className={`inline-block h-2 w-2 shrink-0 translate-y-[1px] ${DOT[l.kind]}`} />
          <span className="sr-only">{l.kind === "model" ? "Model forecast: " : l.kind === "gap" ? "Missing data: " : "Observation: "}</span>
          {l.text}
        </p>
      ))}
    </div>
  );
}
