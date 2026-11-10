"use client";

import { usePathname } from "next/navigation";
import { FrontDoorProvider } from "@/pilot/FrontDoorContext";
import { MotionRoot } from "@/v4/shell/MotionRoot";

/** The pages of the v12 route group (src/app/(neritic)): their shell is the group's own layout. */
export const NERITIC_PATHS = new Set(["/", "/today", "/explore", "/coast", "/research"]);
export const isNeriticPath = (p: string | null) => !!p && NERITIC_PATHS.has(p);

/**
 * Legacy pages (/about, /account, /ask, /sources, /fisheries, legal, /coast/kept) keep the v4 root they were
 * built for: its motion config, its hand-rolled route transitions and the toast. The route group never gets
 * them: its margin uses React's <ViewTransition> and the hand-rolled wrapper would freeze the chart during a
 * slow navigation (research D §4).
 */
export function LegacyChrome({ pilot, children }: { pilot: boolean; children: React.ReactNode }) {
  const pathname = usePathname();
  if (isNeriticPath(pathname)) return <>{children}</>;
  return (
    <FrontDoorProvider pilot={pilot}>
      <MotionRoot>{children}</MotionRoot>
    </FrontDoorProvider>
  );
}
