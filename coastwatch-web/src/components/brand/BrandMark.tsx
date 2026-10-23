import { C2, C2_TONES } from "@/v4/brand/c2";

/**
 * Neritic's mark on the v2 pages (Bloom stations, Fisheries, Sources): C2 "Chart label", the same
 * drawing as the v4 header (src/v4/brand/c2.ts), in its dark master for the navy masthead. The
 * favicon, app icons and sharing image are drawn from the same geometry (scripts/build_brand_assets.mjs).
 */
export function BrandMark({ size = 26, className }: { size?: number; className?: string }) {
  const t = C2_TONES.dark;
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden className={className} data-brand="c2">
      <path d={C2.tile} fill={t.tile} />
      <g stroke={t.stub} strokeWidth={C2.stubWidth}>
        {C2.stubs.map((d) => (
          <path key={d} d={d} />
        ))}
      </g>
      <path transform={C2.nAt} d={C2.n} fill={t.n} />
    </svg>
  );
}

/** The wordmark: "Neritic", one word in one weight, never broken. */
export function Wordmark({ className = "" }: { className?: string }) {
  return <span className={`whitespace-nowrap font-semibold tracking-[-0.01em] ${className}`}>Neritic</span>;
}
