"use client";

import { usePathname } from "next/navigation";
import { FrontDoorProvider } from "@/pilot/FrontDoorContext";
import { MotionRoot } from "@/v4/shell/MotionRoot";

/** The pages of the v12 route group (src/app/(neritic)): their shell is the group's own layout. A page missing here
 *  is wrapped differently by the root, so navigating to it remounts the whole group (the chart included). */
export const NERITIC_PATHS = new Set(["/", "/today", "/explore", "/coast", "/research", "/notices", "/ask", "/account", "/alerts"]);
export const isNeriticPath = (p: string | null) => !!p && NERITIC_PATHS.has(p);

/**
 * /ask, /account and /alerts are listed ahead of their move into the group (v13): until a page has moved it simply
 * renders without the v4 motion root.
 * Legacy pages (/about, /sources, /fisheries, legal, /coast/kept) keep the v4 root they were
 * built for: its motion config, its hand-rolled route transitions and the toast. The route group never gets
 * them: its margin uses React's <ViewTransition> and the hand-rolled wrapper would freeze the chart during a
 * slow navigation (research D §4).
 */
export function LegacyChrome({ pilot, children }: { pilot: boolean; children: React.ReactNode }) {
  const pathname = usePathname();
  // the pilot flag reaches every page (Ask and Your data read it wherever they live); only the legacy pages get
  // the v4 motion root
  return <FrontDoorProvider pilot={pilot}>{isNeriticPath(pathname) ? children : <MotionRoot>{children}</MotionRoot>}</FrontDoorProvider>;
}
