"use client";
import { MotionConfig } from "motion/react";
import { createContext, useContext, useState, type ComponentPropsWithoutRef, type ReactNode } from "react";
import type { ChartMode } from "./mode";
import s from "./Root.module.css";

/** The element overlays portal into, so dialogs, popovers and sheets inherit the scope's chart mode. */
const PortalContainer = createContext<HTMLElement | null>(null);
export const usePortalContainer = () => useContext(PortalContainer);

/**
 * A chart-mode scope. Sets `data-chart-mode` (so the mode's token table applies), the paper ground, the
 * interface face and the focus ring, and makes every Motion animation inside respect reduced motion.
 * Overlays rendered inside portal back into this element, so they keep the mode.
 */
export function ChartModeRoot({ mode, className, children, ...rest }: { mode: ChartMode; children: ReactNode } & ComponentPropsWithoutRef<"div">) {
  const [el, setEl] = useState<HTMLElement | null>(null);
  return (
    <MotionConfig reducedMotion="user">
      <div ref={setEl} data-chart-mode={mode} className={[s.root, className].filter(Boolean).join(" ")} {...rest}>
        <PortalContainer.Provider value={el}>{children}</PortalContainer.Provider>
      </div>
    </MotionConfig>
  );
}
