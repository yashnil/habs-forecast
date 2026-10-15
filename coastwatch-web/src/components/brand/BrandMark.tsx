import { useId } from "react";

/**
 * Neritic's mark: the neritic zone in cross-section. The sea surface, the continental shelf with its
 * break into deep water, the sunlit band of shallow water between them (bright under the surface,
 * fading toward the shelf), and one amber point in it: an observation. Same tile, cyan and amber as before; the favicon, app icons and sharing image are
 * drawn from it (scripts/build_brand_assets.mjs keeps the same paths).
 */
export function BrandMark({ size = 26, className }: { size?: number; className?: string }) {
  const lit = `nm-lit-${useId().replace(/:/g, "")}`;
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden className={className}>
      <rect width="32" height="32" rx="8" fill="#10263e" />
      <defs>
        <linearGradient id={lit} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#6cc6dc" stopOpacity=".42" />
          <stop offset="1" stopColor="#6cc6dc" stopOpacity=".04" />
        </linearGradient>
      </defs>
      {/* the neritic zone: the sunlit water over the shelf */}
      <path d="M6 10H19.5V21H6Z" fill={`url(#${lit})`} />
      {/* sea surface */}
      <path d="M6 10H26" stroke="#6cc6dc" strokeWidth="2" strokeLinecap="round" />
      {/* the shelf, then the break into deep water */}
      <path d="M6 21H19.5L24.5 26.5" fill="none" stroke="#e8eef5" strokeOpacity=".78" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      {/* an observation in the shallow water */}
      <circle cx="12.75" cy="15.5" r="2.2" fill="#f6bb5c" />
    </svg>
  );
}

/** The wordmark: "Neritic", one word in one weight, never broken. */
export function Wordmark({ className = "" }: { className?: string }) {
  return <span className={`whitespace-nowrap font-semibold tracking-[-0.01em] ${className}`}>Neritic</span>;
}
