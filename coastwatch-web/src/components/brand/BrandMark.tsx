/**
 * The Neritic Signal mark: the continental shelf in profile (the neritic zone), the sea surface
 * above it, and a signal rising from the shallow water. Same tile, cyan and amber as before, so the
 * identity stays calm and recognisable; the favicon, app icons and sharing image are drawn from it
 * (scripts/build_brand_assets.mjs).
 */
export function BrandMark({ size = 26, className }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden className={className}>
      <rect width="32" height="32" rx="8" fill="#10263e" />
      {/* sea surface */}
      <path d="M5 10.5c2.6-1.7 5.2-1.7 7.8 0s5.4 1.7 8.2 0 4.6-1.6 6 -.6" fill="none" stroke="#6cc6dc" strokeWidth="2" strokeLinecap="round" />
      {/* the shelf: shallow and flat near the coast, then the break into deep water */}
      <path d="M4.5 23.5h11.2c1.6 0 2.6.5 3.5 1.6l2.6 3.2" fill="none" stroke="#e8eef5" strokeOpacity=".6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      {/* the signal, in the shallow water over the shelf */}
      <circle cx="10.4" cy="18" r="2" fill="#f6bb5c" />
      <path d="M14 15.6a4 4 0 0 1 0 4.8" fill="none" stroke="#f6bb5c" strokeOpacity=".75" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M16.7 14.1a7 7 0 0 1 0 7.8" fill="none" stroke="#f6bb5c" strokeOpacity=".4" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

/** "Neritic Signal" set as one wordmark: the name in two weights, never broken across lines. */
export function Wordmark({ className = "" }: { className?: string }) {
  return (
    <span className={`whitespace-nowrap tracking-tight ${className}`}>
      <span className="font-semibold">Neritic</span> <span className="font-normal">Signal</span>
    </span>
  );
}
