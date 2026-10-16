import { BRAND } from "@/content/brand";

/**
 * PROVISIONAL brand treatment for v4 development. The owner has not approved a final identity
 * (docs/coastwatch/v4/brand/ holds the bathymetric concepts). Every v4 surface takes the mark and
 * wordmark from this one file, so the approved identity replaces it in one place.
 *
 * The placeholder is deliberately plain: a shelf profile (the step from shelf to slope) in a
 * rounded tile, and the name set in the editorial serif.
 */
export function Mark({ size = 28, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className={className} data-brand="provisional">
      <rect x="0.5" y="0.5" width="31" height="31" rx="8" fill="#0d2a37" stroke="rgba(255,255,255,0.14)" />
      <path d="M5 11.5h11.5c2.4 0 3.4 1.2 4.4 3.4L24 22.5H5z" fill="#5fd0e6" opacity="0.92" />
      <path d="M5 11.5h11.5" stroke="#e9f6fa" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function Wordmark({ className = "" }: { className?: string }) {
  return <span className={`font-serif font-semibold tracking-[-0.01em] ${className}`}>{BRAND.name}</span>;
}

export function Lockup({ size = 28, className = "", wordClass = "text-[19px]" }: { size?: number; className?: string; wordClass?: string }) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${className}`}>
      <Mark size={size} />
      <Wordmark className={wordClass} />
    </span>
  );
}
