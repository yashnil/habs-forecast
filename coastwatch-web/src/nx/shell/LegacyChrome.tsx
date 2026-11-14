"use client";

import { usePathname } from "next/navigation";
import { FrontDoorProvider } from "@/pilot/FrontDoorContext";
import { MotionRoot } from "@/v4/shell/MotionRoot";

/** The pages of the v12 route group (src/app/(neritic)): their shell is the group's own layout. A page missing here
 *  is wrapped differently by the root, so navigating to it remounts the whole group (the chart included). */
export const NERITIC_PATHS = new Set([
  "/", "/today", "/explore", "/coast", "/coast/trips", "/research", "/notices", "/ask", "/account", "/alerts",
  // v15.1: the informational pages moved into the group, on the design system's page layout (src/nx/pages)
  "/about", "/sources", "/privacy", "/terms", "/sms-terms", "/fisheries", "/coast/kept", "/account/pilot",
]);
/** The pages still outside the group: the design-system catalog (its own review surface). */
export const LEGACY_PATHS = ["/system"];
/** A path of the group, or an address no route claims (the root not-found draws the same shell: src/nx/pages/StandaloneShell). */
export const isNeriticPath = (p: string | null) => !!p && (NERITIC_PATHS.has(p) || !LEGACY_PATHS.some((l) => p === l || p.startsWith(`${l}/`)));

/**
 * v15.1: every reader-facing page is in the group now; About, Sources, the legal pages, Fisheries, kept briefings and
 * the not-found page render on the design system's informational layout (src/nx/pages, docs/coastwatch/v15.1/routes.md).
 * Only /system (the component catalog) keeps the v4 root: its motion config, its hand-rolled route transitions and
 * the toast. The route group never gets them: its margin uses React's <ViewTransition> and the hand-rolled wrapper
 * would freeze the chart during a slow navigation (research D §4).
 */
export function LegacyChrome({ pilot, children }: { pilot: boolean; children: React.ReactNode }) {
  const pathname = usePathname();
  // the pilot flag reaches every page (Ask and Your data read it wherever they live); only the legacy pages get
  // the v4 motion root
  return <FrontDoorProvider pilot={pilot}>{isNeriticPath(pathname) ? children : <MotionRoot>{children}</MotionRoot>}</FrontDoorProvider>;
}
