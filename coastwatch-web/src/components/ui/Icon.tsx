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
  info: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5.2M12 7.8v.1" />
    </>
  ),
  close: <path d="M6 6l12 12M18 6 6 18" />,
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, className = "h-[18px] w-[18px]" }: { name: IconName; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className={`shrink-0 ${className}`} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      {PATHS[name]}
    </svg>
  );
}
