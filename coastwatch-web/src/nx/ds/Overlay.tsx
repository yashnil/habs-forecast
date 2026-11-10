"use client";
import * as RD from "@radix-ui/react-dialog";
import * as RP from "@radix-ui/react-popover";
import * as RT from "@radix-ui/react-tooltip";
import { AnimatePresence, motion } from "motion/react";
import { useState, type ReactElement, type ReactNode } from "react";
import { Icon } from "./Icon";
import { marginSwap, register, registerFrom, registerTo, sheet } from "./motion";
import { usePortalContainer } from "./Root";
import { usePhone } from "./useMedia";
import { cx } from "./util";
import s from "./Overlay.module.css";

function useOpen(open: boolean | undefined, onOpenChange: ((o: boolean) => void) | undefined, initial = false) {
  const [inner, setInner] = useState(initial);
  const value = open ?? inner;
  const set = (o: boolean) => {
    if (open === undefined) setInner(o);
    onOpenChange?.(o);
  };
  return [value, set] as const;
}

export interface DialogProps {
  title: ReactNode;
  /** One sentence under the title; also the dialog's accessible description. */
  description?: ReactNode;
  /** "center" for a decision; "side" for a sheet of detail (right on desktop, bottom on phone). */
  placement?: "center" | "side";
  trigger?: ReactElement;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  children?: ReactNode;
  footer?: ReactNode;
  className?: string;
}

/**
 * A Radix dialog dressed as paper: 1.5 px ink rule, no shadow, no radius. The page behind fades under a
 * paper halo rather than a dark scrim. Focus is trapped and returned; Escape closes.
 */
export function Dialog({ title, description, placement = "center", trigger, open, defaultOpen, onOpenChange, children, footer, className }: DialogProps) {
  const [isOpen, setOpen] = useOpen(open, onOpenChange, defaultOpen);
  const container = usePortalContainer();
  const phone = usePhone();
  const side = placement === "side";
  const from = side ? (phone ? { opacity: 0, y: 24 } : { opacity: 0, x: 24 }) : registerFrom;
  const to = side ? { opacity: 1, x: 0, y: 0 } : registerTo;
  return (
    <RD.Root open={isOpen} onOpenChange={setOpen}>
      {trigger ? <RD.Trigger asChild>{trigger}</RD.Trigger> : null}
      <AnimatePresence>
        {isOpen ? (
          <RD.Portal forceMount container={container ?? undefined}>
            <RD.Overlay asChild forceMount>
              <motion.div className={s.scrim} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={marginSwap} />
            </RD.Overlay>
            <RD.Content asChild forceMount>
              <motion.div
                className={cx(s.dialog, side ? s.side : s.center, className)}
                initial={from}
                animate={to}
                exit={from}
                transition={side ? sheet : register}
              >
                <div className={s.head}>
                  <RD.Title className={s.title}>{title}</RD.Title>
                  <RD.Close className={s.close} aria-label="Close">
                    <Icon name="close" />
                  </RD.Close>
                </div>
                {description ? <RD.Description className={s.description}>{description}</RD.Description> : <RD.Description className={s.vhDesc}>{typeof title === "string" ? title : "Details"}</RD.Description>}
                <div className={s.body}>{children}</div>
                {footer ? <div className={s.footer}>{footer}</div> : null}
              </motion.div>
            </RD.Content>
          </RD.Portal>
        ) : null}
      </AnimatePresence>
    </RD.Root>
  );
}

export interface PopoverProps {
  trigger: ReactElement;
  children: ReactNode;
  /** Accessible name when the content has no heading. */
  label?: string;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
}

/** A small ruled sheet next to its control (port switcher, layer list). */
export function Popover({ trigger, children, label, side = "bottom", align = "start", open, onOpenChange, className }: PopoverProps) {
  const [isOpen, setOpen] = useOpen(open, onOpenChange);
  const container = usePortalContainer();
  return (
    <RP.Root open={isOpen} onOpenChange={setOpen}>
      <RP.Trigger asChild>{trigger}</RP.Trigger>
      <AnimatePresence>
        {isOpen ? (
          <RP.Portal forceMount container={container ?? undefined}>
            <RP.Content asChild forceMount side={side} align={align} sideOffset={8} collisionPadding={16} aria-label={label}>
              <motion.div className={cx(s.popover, className)} initial={registerFrom} animate={registerTo} exit={registerFrom} transition={register}>
                {children}
              </motion.div>
            </RP.Content>
          </RP.Portal>
        ) : null}
      </AnimatePresence>
    </RP.Root>
  );
}

export interface TooltipProps {
  /** Supplementary words only: a tooltip never carries the only copy of anything a reader needs. */
  content: ReactNode;
  children: ReactElement;
  side?: "top" | "right" | "bottom" | "left";
}

/** Radix tooltip, 14 px, ruled. Needs a TooltipProvider above (one per app). */
export function Tooltip({ content, children, side = "top" }: TooltipProps) {
  const [isOpen, setOpen] = useState(false);
  const container = usePortalContainer();
  return (
    <RT.Root open={isOpen} onOpenChange={setOpen}>
      <RT.Trigger asChild>{children}</RT.Trigger>
      <AnimatePresence>
        {isOpen ? (
          <RT.Portal forceMount container={container ?? undefined}>
            <RT.Content asChild forceMount side={side} sideOffset={6} collisionPadding={16}>
              <motion.div className={s.tooltip} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={marginSwap}>
                {content}
              </motion.div>
            </RT.Content>
          </RT.Portal>
        ) : null}
      </AnimatePresence>
    </RT.Root>
  );
}

export function TooltipProvider({ children }: { children: ReactNode }) {
  return (
    <RT.Provider delayDuration={300} skipDelayDuration={200}>
      {children}
    </RT.Provider>
  );
}
