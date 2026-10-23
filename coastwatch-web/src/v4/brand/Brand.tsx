import { useId } from "react";
import { BRAND } from "@/content/brand";
import { LOCKUP_WIDTH, MARK_GAP, MARK_INDEX_DEPTH, MARK_TIERS, MARK_TILE, MARK_TONES, MARK_WORD, strokeOf, tierFor, type MarkTier } from "./mark";

/**
 * Neritic's identity: the contour mark (src/v4/brand/mark.ts, drawn from Monterey Canyon's
 * isobaths) and the Source Serif wordmark. Every v4 surface takes them from this one file. The
 * colours follow the environment through the --nx-mark-* tokens in theme.css: the light master on
 * the reading pages (`env-chart`), the dark master on the map's instrument surface
 * (`env-instrument`); the light master is the fallback anywhere else.
 */
const L = MARK_TONES.light;
const v = (name: string, fallback: string) => `var(--nx-mark-${name}, ${fallback})`;

function ContourSymbol({ tier }: { tier: MarkTier }) {
  const clip = `nxm-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  return (
    <>
      <defs>
        <clipPath id={clip}>
          <path d={MARK_TILE} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clip})`}>
        <path d={MARK_TILE} fill={v("tile", L.tile)} />
        <g fill="none" strokeLinecap="round">
          {MARK_TIERS[tier].lines.map((l, i) => (
            <path key={i} d={l.d} stroke={l.depth === MARK_INDEX_DEPTH ? v("index", L.index) : v("line", L.line)} strokeWidth={strokeOf(tier, l.depth)} />
          ))}
        </g>
      </g>
    </>
  );
}

export function Mark({ size = 28, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden="true" className={`shrink-0 ${className}`} data-brand="contour">
      <ContourSymbol tier={tierFor(size)} />
    </svg>
  );
}

/** The drawn wordmark (Source Serif 4 Semibold, outlined); text and labels keep "Neritic". */
export function Wordmark({ height = 28, className = "" }: { height?: number; className?: string }) {
  const w = MARK_WORD.width + 2;
  return (
    <svg height={height} width={(w / 48) * height} viewBox={`-1 0 ${w} 48`} role="img" aria-label={BRAND.name} className={`shrink-0 ${className}`}>
      <path transform={`translate(0 ${MARK_WORD.baseline})`} d={MARK_WORD.d} fill={v("word", L.word)} />
    </svg>
  );
}

/** Mark and wordmark as one drawing, spaced as in the master lockup. */
export function Lockup({ size = 28, className = "" }: { size?: number; className?: string }) {
  return (
    <svg height={size} width={(LOCKUP_WIDTH / 48) * size} viewBox={`0 0 ${LOCKUP_WIDTH} 48`} aria-hidden="true" className={`shrink-0 ${className}`} data-brand="contour">
      <ContourSymbol tier={tierFor(size)} />
      <path transform={`translate(${48 + MARK_GAP} ${MARK_WORD.baseline})`} d={MARK_WORD.d} fill={v("word", L.word)} />
    </svg>
  );
}
