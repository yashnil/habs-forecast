/** v4 line icons: 24 px grid, 1.7 px stroke, round joins, currentColor. Always decorative. */
const P: Record<string, React.ReactNode> = {
  layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></>,
  search: <><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" /></>,
  close: <path d="M6 6l12 12M18 6 6 18" />,
  back: <path d="M15 5 8 12l7 7" />,
  chevronDown: <path d="m6 9 6 6 6-6" />,
  chevronRight: <path d="m9 6 6 6-6 6" />,
  plus: <path d="M12 5v14M5 12h14" />,
  minus: <path d="M5 12h14" />,
  locate: <><circle cx="12" cy="12" r="4" /><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3" /></>,
  share: <><path d="M12 3v12" /><path d="m7.5 7.5 4.5-4.5 4.5 4.5" /><path d="M5 12v7h14v-7" /></>,
  area: <><path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4" /><path d="M9 12h6M12 9v6" /></>,
  clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></>,
  offline: <><path d="M3 3l18 18" /><path d="M8.5 8.6A7 7 0 0 0 6 13.5M12 6.5a7 7 0 0 1 6.9 5.8M9.6 16.4a3.6 3.6 0 0 1 4.4-.5" /><circle cx="12" cy="19.5" r="0.9" fill="currentColor" stroke="none" /></>,
  online: <><path d="M3.5 9.5a12 12 0 0 1 17 0M6.5 12.6a7.5 7.5 0 0 1 11 0M9.5 15.7a3.4 3.4 0 0 1 5 0" /><circle cx="12" cy="19" r="0.9" fill="currentColor" stroke="none" /></>,
  refresh: <><path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3" /><path d="M19.5 4.5v4.2h-4.2" /></>,
  download: <><path d="M12 4v11" /><path d="m7.5 10.5 4.5 4.5 4.5-4.5" /><path d="M5 19.5h14" /></>,
  check: <path d="m5 12.5 4.5 4.5L19 7.5" />,
  alert: <><path d="M12 4 2.8 19.5h18.4L12 4Z" /><path d="M12 10v4.5" /><circle cx="12" cy="17" r="0.9" fill="currentColor" stroke="none" /></>,
  info: <><circle cx="12" cy="12" r="8.5" /><path d="M12 11v5.5" /><circle cx="12" cy="7.8" r="0.9" fill="currentColor" stroke="none" /></>,
  shield: <path d="M12 3 4.8 6v5.4c0 4.4 3.1 7.9 7.2 9.4 4.1-1.5 7.2-5 7.2-9.4V6L12 3Z" />,
  station: <><circle cx="12" cy="9" r="3" /><path d="M12 12v9M8 21h8" /></>,
  cell: <><rect x="5" y="5" width="14" height="14" rx="1" /><path d="M5 12h14M12 5v14" opacity="0.5" /></>,
  harbor: <><circle cx="12" cy="5.5" r="2" /><path d="M12 7.5V20M7 11h10M4.5 14.5c1.2 3.4 4.2 5.5 7.5 5.5s6.3-2.1 7.5-5.5" /></>,
  chart: <><path d="M4 4v16h16" /><path d="m7 15 4-5 3 3 5-7" /></>,
  book: <><path d="M5 4.5h9.5a3 3 0 0 1 3 3V20H8a3 3 0 0 1-3-3V4.5Z" /><path d="M5 17a3 3 0 0 1 3-3h9.5" /></>,
  compass: <><circle cx="12" cy="12" r="8.5" /><path d="m15.5 8.5-2 5-5 2 2-5 5-2Z" /></>,
  research: <><circle cx="10.5" cy="10.5" r="6" /><path d="m15 15 5 5" /><path d="M7.5 12 9.5 9.5l2 2 2-3" /></>,
  external: <><path d="M14 4h6v6" /><path d="M20 4 11 13" /><path d="M18 14v5H5V6h5" /></>,
  link: <><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" /><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" /></>,
  keep: <><path d="M6.5 3.5h11v17l-5.5-4-5.5 4v-17Z" /></>,
  map: <><path d="M9 4 3 6.5v13.5l6-2.5 6 2.5 6-2.5V4l-6 2.5L9 4Z" /><path d="M9 4v13.5M15 6.5V20" /></>,
  more: <><circle cx="5.5" cy="12" r="1.3" fill="currentColor" stroke="none" /><circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none" /><circle cx="18.5" cy="12" r="1.3" fill="currentColor" stroke="none" /></>,
  keyboard: <><rect x="3" y="6.5" width="18" height="11" rx="2" /><path d="M7 10h.01M10 10h.01M13 10h.01M16 10h.01M8 14h8" /></>,
};

export type IconName = keyof typeof P;

/** True when the caller's classes give the icon a height or width of its own. */
const SIZED = /(^|\s)(?:[a-z0-9-]+:)*(?:h|w|size)-/;

/** Always sized: 20 px unless the caller names a size (a caller class with only a colour once left the
 *  SVG at its intrinsic 300 × 150 px), and never squeezed by a flex row. */
export function iconClass(className = ""): string {
  return `shrink-0 ${SIZED.test(className) ? "" : "h-5 w-5 "}${className}`.trim();
}

export function Icon({ name, className = "", strokeWidth = 1.7 }: { name: IconName | string; className?: string; strokeWidth?: number }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" className={iconClass(className)} aria-hidden="true" focusable="false">
      {P[name] ?? P.info}
    </svg>
  );
}
