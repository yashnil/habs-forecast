import { BRAND } from "@/content/brand";
import { C2, C2_TONES } from "./c2";

/**
 * Neritic's identity, C2 "Chart label" (src/v4/brand/c2.ts). Every v4 surface takes the mark and
 * wordmark from this one file. The tile follows its environment: the light master on the reading
 * pages (`env-chart`), the dark master on the map's instrument surface (`env-instrument`), through the
 * --nx-mark-* tokens in theme.css; the light master is the fallback anywhere else.
 */
const L = C2_TONES.light;
const v = (name: string, fallback: string) => `var(--nx-mark-${name}, ${fallback})`;

// No clip path: the contour stubs end flush with the tile on its straight edges (y 23-26 of 48),
// so every instance is self-contained (no shared ids) and the lockup needs no crop either.
function C2Symbol() {
  return (
    <>
      <path d={C2.tile} fill={v("tile", L.tile)} />
      <g stroke={v("stub", L.stub)} strokeWidth={C2.stubWidth}>
        {C2.stubs.map((d) => (
          <path key={d} d={d} />
        ))}
      </g>
      <path transform={C2.nAt} d={C2.n} fill={v("n", L.n)} />
    </>
  );
}

export function Mark({ size = 28, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden="true" className={`shrink-0 ${className}`} data-brand="c2">
      <C2Symbol />
    </svg>
  );
}

/** The drawn wordmark (outlined italic lowercase, as water names are set on charts); text and labels keep "Neritic". */
export function Wordmark({ height = 28, className = "" }: { height?: number; className?: string }) {
  const w = C2.lockupWidth - 58;
  return (
    <svg height={height} width={(w / 48) * height} viewBox={`58 0 ${w} 48`} role="img" aria-label={BRAND.name} className={`shrink-0 ${className}`}>
      <path transform={C2.wordAt} d={C2.word} fill={v("word", L.word)} />
    </svg>
  );
}

/** Mark and wordmark as one drawing, spaced as in the master lockup. */
export function Lockup({ size = 28, className = "" }: { size?: number; className?: string }) {
  return (
    <svg height={size} width={(C2.lockupWidth / 48) * size} viewBox={`0 0 ${C2.lockupWidth} 48`} aria-hidden="true" className={`shrink-0 ${className}`} data-brand="c2">
      <C2Symbol />
      <path transform={C2.wordAt} d={C2.word} fill={v("word", L.word)} />
    </svg>
  );
}
