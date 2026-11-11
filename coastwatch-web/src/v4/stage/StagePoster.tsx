import { MARK_FIELD, MARK_INDEX_DEPTH } from "@/v4/brand/mark.geometry";

/** A line's opacity by depth: the coast and the 200 m index carry the chart; the deep falls away. */
export function posterOpacity(depth: number): number {
  if (depth === 0) return 0.42;
  if (depth === MARK_INDEX_DEPTH) return 0.56;
  if (depth <= 150) return 0.28;
  if (depth <= 600) return 0.22;
  return Math.max(0.1, 0.18 - ((depth - 800) / 1200) * 0.08);
}

const H = MARK_FIELD.half;

/**
 * The stage before (or without) the map: the real Monterey Canyon isobaths (the brand's own geometry, from
 * NOAA's Coastal Relief Model) on the deep ground. Inline SVG, zero requests, no glow. It is the brand's
 * ground, never presented as the reader's waters.
 */
export function StagePoster() {
  return (
    <svg className="nx-stage-poster" viewBox={`${24 - H} ${24 - H} ${2 * H} ${2 * H}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true" data-testid="stage-poster">
      <g fill="none" stroke="currentColor" strokeLinecap="round">
        {MARK_FIELD.lines.map((l, i) => (
          <path key={i} d={l.d} strokeOpacity={posterOpacity(l.depth)} data-index={l.depth === MARK_INDEX_DEPTH ? "" : undefined} />
        ))}
      </g>
    </svg>
  );
}
