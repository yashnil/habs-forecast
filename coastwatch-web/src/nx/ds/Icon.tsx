import type { ReactNode, SVGProps } from "react";

/**
 * Neritic icons (spec section 7). Drawn for this product on a 24 grid at 1.5 px stroke, from chart
 * symbology. Ink only (currentColor). Decorative by default (aria-hidden); pass `label` only for a
 * universal control that has no word beside it (close, zoom), and the icon becomes an image with a name.
 */
export type IconName =
  | "port"
  | "buoy"
  | "testArea"
  | "tideStation"
  | "radar"
  | "zoneLine"
  | "search"
  | "layers"
  | "time"
  | "close"
  | "chevron"
  | "share"
  | "offline"
  | "alert"
  | "plus"
  | "minus"
  | "menu"
  | "recentre";

const PATHS: Record<IconName, ReactNode> = {
  // a port on the chart: circled dot
  port: (
    <>
      <circle cx="12" cy="12" r="7.25" />
      <circle cx="12" cy="12" r="2.25" fill="currentColor" stroke="none" />
    </>
  ),
  // a pillar buoy standing on its position circle
  buoy: (
    <>
      <path d="M9.25 17.25 11.4 6.1a.6.6 0 0 1 1.18 0l2.17 11.15" />
      <path d="M8.5 13.25h7" />
      <circle cx="12" cy="19.25" r="1.75" />
      <path d="M3.75 19.25h4.5M15.75 19.25h4.5" />
    </>
  ),
  // a sampling area: dashed boundary, the sample at its centre
  testArea: (
    <>
      <path d="M4.75 8.25v-3.5h3.5M15.75 4.75h3.5v3.5M19.25 15.75v3.5h-3.5M8.25 19.25h-3.5v-3.5" />
      <rect x="9.75" y="9.75" width="4.5" height="4.5" fill="currentColor" stroke="none" />
    </>
  ),
  // a tide staff over the water line
  tideStation: (
    <>
      <path d="M12 3.75v12.5" />
      <path d="M12 6.25h3.25M12 9.25h2M12 12.25h3.25" />
      <path d="M3.75 18.25c1.4-1.25 2.75-1.25 4.125 0s2.75 1.25 4.125 0 2.75-1.25 4.125 0 2.75 1.25 4.125 0" />
    </>
  ),
  // shore radar: the site and its coverage arcs
  radar: (
    <>
      <circle cx="12" cy="18.5" r="1.75" fill="currentColor" stroke="none" />
      <path d="M8.1 14.6a5.5 5.5 0 0 1 7.8 0" />
      <path d="M5.3 11.8a9.5 9.5 0 0 1 13.4 0" />
      <path d="M2.75 8.75a13.1 13.1 0 0 1 18.5 0" />
    </>
  ),
  // a regulatory zone line: the chart's T-dash
  zoneLine: (
    <>
      <path d="M2.75 9.25h4.5M9.75 9.25h4.5M16.75 9.25h4.5" />
      <path d="M5 9.25v6M12 9.25v6M19 9.25v6" />
    </>
  ),
  search: (
    <>
      <circle cx="10.5" cy="10.5" r="6.25" />
      <path d="m15.1 15.1 5.15 5.15" />
    </>
  ),
  // stacked sheets of a chart
  layers: (
    <>
      <path d="m12 4.25 8.25 4.25L12 12.75 3.75 8.5 12 4.25Z" />
      <path d="m3.75 12.25 8.25 4.25 8.25-4.25" />
      <path d="m3.75 16 8.25 4.25L20.25 16" />
    </>
  ),
  time: (
    <>
      <circle cx="12" cy="12" r="8.25" />
      <path d="M12 7.25V12l3.25 2" />
    </>
  ),
  close: <path d="m6.25 6.25 11.5 11.5M17.75 6.25l-11.5 11.5" />,
  chevron: <path d="m6.75 9.5 5.25 5.25 5.25-5.25" />,
  share: (
    <>
      <path d="M12 14.25V3.75M8.25 7.5 12 3.75l3.75 3.75" />
      <path d="M8.25 10.75h-3.5v9.5h14.5v-9.5h-3.5" />
    </>
  ),
  // no connection: the signal arcs, struck through
  offline: (
    <>
      <circle cx="12" cy="18.75" r="1.5" fill="currentColor" stroke="none" />
      <path d="M8.9 15.3a4.6 4.6 0 0 1 6.2 0" />
      <path d="M5.7 12.2a9 9 0 0 1 9.6-1.8" />
      <path d="M2.75 9.1a13.3 13.3 0 0 1 8.1-3.3" />
      <path d="m3.75 3.75 16.5 16.5" />
    </>
  ),
  // a bell, for the Alerts destination
  alert: (
    <>
      <path d="M6.25 16.25V11a5.75 5.75 0 0 1 11.5 0v5.25l1.5 1.5H4.75l1.5-1.5Z" />
      <path d="M10 20.25h4" />
    </>
  ),
  plus: <path d="M12 5.25v13.5M5.25 12h13.5" />,
  minus: <path d="M5.25 12h13.5" />,
  menu: <path d="M4.75 7.25h14.5M4.75 12h14.5M4.75 16.75h14.5" />,
  // back to the page's place: a ring with four ticks
  recentre: (
    <>
      <circle cx="12" cy="12" r="5.75" />
      <path d="M12 2.75v3.5M12 17.75v3.5M2.75 12h3.5M17.75 12h3.5" />
    </>
  ),
};

export const ICON_NAMES = Object.keys(PATHS) as IconName[];

export interface IconProps extends Omit<SVGProps<SVGSVGElement>, "name"> {
  name: IconName;
  /** Rendered size in px; the drawing stays on its 24 grid. */
  size?: number;
  /** Only for a universal control with no word beside it. */
  label?: string;
  /** Chevron direction. */
  direction?: "down" | "up" | "left" | "right";
}

const ROT = { down: 0, up: 180, left: 90, right: -90 } as const;

export function Icon({ name, size = 24, label, direction, style, ...rest }: IconProps) {
  const rot = direction ? ROT[direction] : 0;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="butt"
      strokeLinejoin="miter"
      aria-hidden={label ? undefined : true}
      role={label ? "img" : undefined}
      aria-label={label}
      focusable="false"
      style={{ flex: "none", transform: rot ? `rotate(${rot}deg)` : undefined, ...style }}
      {...rest}
    >
      {PATHS[name]}
    </svg>
  );
}
