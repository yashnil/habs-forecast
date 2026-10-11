import type { Freshness } from "@/lib/freshness";

/** Freshness as a glyph and a word (spec §2.2): filled = current, half = stale, ring = historical.
 *  Colour is never the only signal. */
const WORD = { current: "Current", stale: "Stale", historical: "Not current", unavailable: "Unavailable" } as const;
const COLOR = { current: "#17704a", stale: "#86590a", historical: "#5c6470", unavailable: "#5c6470" } as const;

export function FreshGlyph({ state, size = 9 }: { state: Freshness["state"]; size?: number }) {
  const c = COLOR[state];
  const r = size / 2 - 1;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden className="shrink-0">
      {state === "current" && <circle cx={size / 2} cy={size / 2} r={r} fill={c} />}
      {state === "stale" && (
        <>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={c} strokeWidth="1.4" />
          <path d={`M${size / 2},${size / 2 - r} A${r},${r} 0 0 0 ${size / 2},${size / 2 + r} Z`} fill={c} />
        </>
      )}
      {(state === "historical" || state === "unavailable") && <circle cx={size / 2} cy={size / 2} r={r - 0.4} fill="none" stroke={c} strokeWidth="1.4" />}
    </svg>
  );
}

export function FreshDot({ f, showWord = true, className = "" }: { f: Freshness | null; showWord?: boolean; className?: string }) {
  if (!f) return null;
  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`} data-state={f.state} data-testid="fresh">
      <FreshGlyph state={f.state} />
      {showWord ? <span style={{ color: COLOR[f.state] }} className="font-medium">{WORD[f.state]}</span> : <span className="sr-only">{WORD[f.state]}</span>}
    </span>
  );
}

export const freshWord = (s: Freshness["state"]) => WORD[s];
