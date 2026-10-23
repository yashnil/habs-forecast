import { useId } from "react";
import { MARK_INDEX_DEPTH, MARK_TIERS, MARK_TILE, MARK_TONES, strokeOf, tierFor } from "@/v4/brand/mark";

/**
 * Neritic's mark on the v2 pages (Bloom stations, Fisheries, Sources): the contour mark, the same
 * drawing as the v4 header (src/v4/brand/mark.ts), in its dark master for the navy masthead. The
 * favicon, app icons and sharing images are drawn from the same geometry (scripts/build_brand_assets.mjs).
 */
export function BrandMark({ size = 26, className }: { size?: number; className?: string }) {
  const t = MARK_TONES.dark;
  const tier = tierFor(size);
  const clip = `nxb-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden className={className} data-brand="contour">
      <defs>
        <clipPath id={clip}>
          <path d={MARK_TILE} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clip})`}>
        <path d={MARK_TILE} fill={t.tile} />
        <g fill="none" strokeLinecap="round">
          {MARK_TIERS[tier].lines.map((l, i) => (
            <path key={i} d={l.d} stroke={l.depth === MARK_INDEX_DEPTH ? t.index : t.line} strokeWidth={strokeOf(tier, l.depth)} />
          ))}
        </g>
      </g>
    </svg>
  );
}

/** The wordmark: "Neritic", one word in one weight, never broken. */
export function Wordmark({ className = "" }: { className?: string }) {
  return <span className={`whitespace-nowrap font-semibold tracking-[-0.01em] ${className}`}>Neritic</span>;
}
