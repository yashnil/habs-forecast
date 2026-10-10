"use client";

/**
 * What the map shows and when, in one glance, on the map itself: one line per layer drawn
 * (observation or model), with its observation or valid time. The dock has the detail; this
 * keeps the time visible when the dock is collapsed (phones) or scrolled.
 */
export type StampLine = { kind: "observation" | "model" | "gap"; text: string; testid: string };

export function MapStamp({ lines, className = "", style }: { lines: StampLine[]; className?: string; style?: React.CSSProperties }) {
  if (!lines.length) return null;
  return (
    <div data-testid="map-stamp" style={style} className={`pointer-events-none flex flex-col gap-0.5 rounded-lg bg-navy-900/88 px-2.5 py-1.5 text-[12px] leading-snug text-white shadow-[0_2px_10px_rgba(6,17,30,0.35)] backdrop-blur-[2px] ${className}`}>
      {lines.map((l) => (
        <p key={l.testid} data-testid={l.testid} className="flex items-baseline gap-1.5">
          <span aria-hidden className={`inline-block h-2 w-2 shrink-0 translate-y-[1px] ${l.kind === "gap" ? "border border-dashed border-[#f6bb5c] bg-transparent" : "rounded-full"} ${l.kind === "model" ? "bg-[#b9a6f5]" : l.kind === "observation" ? "bg-[#7fd4c1]" : ""}`} />
          <span className="sr-only">{l.kind === "model" ? "Model forecast: " : l.kind === "gap" ? "Missing data: " : "Observation: "}</span>
          {l.text}
        </p>
      ))}
    </div>
  );
}
