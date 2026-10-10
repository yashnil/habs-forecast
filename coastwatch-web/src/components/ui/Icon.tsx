/** Line icons for the shell (24 px grid, 1.6 px stroke, currentColor). Decorative: always aria-hidden. */
const PATHS = {
  map: (
    <>
      <path d="M9 4 3 6.5v13.5l6-2.5 6 2.5 6-2.5V4l-6 2.5L9 4Z" />
      <path d="M9 4v13.5M15 6.5V20" />
    </>
  ),
  bloom: (
    <>
      <circle cx="6" cy="17" r="2.2" />
      <circle cx="12" cy="12" r="2.2" />
      <circle cx="18" cy="7" r="2.2" />
      <path d="m7.6 15.4 2.8-1.8M13.6 10.4l2.8-1.8" />
    </>
  ),
  fish: (
    <>
      <path d="M3 12c3-4.5 7.5-6 11-4.5 2.2 1 3.8 2.7 4.8 4.5-1 1.8-2.6 3.5-4.8 4.5C10.5 18 6 16.5 3 12Z" />
      <path d="m18.8 12 2.2-3v6l-2.2-3Z" />
    </>
  ),
  shield: (
    <>
      <path d="M12 3 4.5 6v5.5c0 4.5 3.2 8 7.5 9.5 4.3-1.5 7.5-5 7.5-9.5V6L12 3Z" />
      <path d="M12 8v5M12 16.2v.1" />
    </>
  ),
  close: <path d="M6 6l12 12M18 6 6 18" />,
  // a model: observed line, then a dashed projection
  forecast: (
    <>
      <path d="M3 17.5 7 13l3.5 2.5L14 11" />
      <path d="M14 11l3.2-2.6L21 9" strokeDasharray="2 2.4" />
      <path d="M3 21h18" />
    </>
  ),
  satellite: (
    <>
      <path d="m10.5 8.5 5 5-2.5 2.5-5-5 2.5-2.5Z" />
      <path d="m8 11-4.5-4.5 2.5-2.5L10.5 8.5M15.5 13.5l4.5 4.5-2.5 2.5L13 16" />
      <path d="M5 19c0-1.7 1.3-3 3-3M3 19c0-2.8 2.2-5 5-5" />
    </>
  ),
  currents: (
    <>
      <path d="M3 8c2.5-2 5-2 7.5 0s5 2 7.5 0" />
      <path d="M3 14c2.5-2 5-2 7.5 0s5 2 7.5 0" />
      <path d="m16.5 5.6 1.8 2.3-2.4 1.6M16.5 11.6l1.8 2.3-2.4 1.6" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6.5" />
      <path d="m20 20-4.2-4.2" />
    </>
  ),
  pin: (
    <>
      <path d="M12 21s-6.5-5.6-6.5-11a6.5 6.5 0 0 1 13 0c0 5.4-6.5 11-6.5 11Z" />
      <circle cx="12" cy="10" r="2.3" />
    </>
  ),
  layers: (
    <>
      <path d="m12 4 8.5 4.5L12 13 3.5 8.5 12 4Z" />
      <path d="m3.5 12.5 8.5 4.5 8.5-4.5M3.5 16.5 12 21l8.5-4.5" />
    </>
  ),
  chevron: <path d="m9 6 6 6-6 6" />,
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, className = "h-[18px] w-[18px]" }: { name: IconName; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className={`shrink-0 ${className}`} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      {PATHS[name]}
    </svg>
  );
}
