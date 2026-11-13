"use client";

import { animate } from "motion/react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { noteTicketLeaving, whenFlightNear } from "@/nx/chart/flight";
import { ticketOut } from "../motion";
import { PHONE_QUERY } from "../useMedia";

/** The ghost layer the workspace keeps over the stage: a leaving object's copy shrinks there after the page has gone. */
export const GHOST_LAYER_ID = "nx-stage-ghosts";

export const isPhoneNow = () => typeof window !== "undefined" && window.matchMedia(PHONE_QUERY).matches;

/**
 * An object on the stage arriving with the camera: false until the flight now under way is 62 % done (at once when
 * the camera is still), then true. `extraDelay` (s) puts a dock 70 ms behind its ticket.
 */
export function useArrival(extraDelay = 0): boolean {
  const [here, setHere] = useState(false);
  useEffect(() => {
    let live = true;
    whenFlightNear().then(() => {
      if (!live) return;
      if (extraDelay > 0) setTimeout(() => live && setHere(true), extraDelay * 1000);
      else setHere(true);
    });
    return () => {
      live = false;
    };
  }, [extraDelay]);
  return here;
}

/**
 * When the object leaves (its page changed, or its place did), a copy of it shrinks toward its anchor over 170 ms on
 * the ghost layer, opaque until its last frames, while the camera waits for it (chart/flight). Desktop only: on a
 * phone the sheet's content simply changes.
 */
export function useLeaveGhost<T extends HTMLElement>(ref: React.RefObject<T | null>, scaleTo = 0.94) {
  // the source's box in the stage, measured at each commit while it is laid out as a stage object (offsets ignore the
  // arrival's transforms). By the time it leaves, the old page's layout marker may already be gone and the stage
  // laid out otherwise, so the leaving copy never trusts a measurement taken then.
  const last = useRef<{ node: HTMLElement; left: number; top: number; width: number; height: number } | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || isPhoneNow()) return;
    last.current = { node: el, left: el.offsetLeft, top: el.offsetTop, width: el.offsetWidth, height: el.offsetHeight };
  });
  useLayoutEffect(
    () => () => {
      const box = last.current;
      const src = box?.node;
      const layer = document.getElementById(GHOST_LAYER_ID);
      if (!box || !src || !layer || isPhoneNow() || !box.width || !box.height) return;
      // a ticket follows its place with inline left/top (the camera may have moved since the last commit)
      const left = src.style.left ? parseFloat(src.style.left) : box.left;
      const top = src.style.top ? parseFloat(src.style.top) : box.top;
      const ghost = src.cloneNode(true) as HTMLElement;
      // a leaving copy is decoration: it answers for nothing on the page (no ids, test ids or the desk header's hook)
      for (const n of [ghost, ...ghost.querySelectorAll<HTMLElement>("[id], [data-testid], [data-nx-desk-header]")]) {
        n.removeAttribute("id");
        n.removeAttribute("data-testid");
        n.removeAttribute("data-nx-desk-header");
      }
      ghost.setAttribute("aria-hidden", "true");
      ghost.setAttribute("inert", "");
      // exactly the source's box, whatever its class would compute in the ghost layer
      Object.assign(ghost.style, {
        position: "absolute",
        inset: "auto",
        left: `${left}px`,
        top: `${top}px`,
        width: `${box.width}px`,
        height: `${box.height}px`,
        minWidth: "0",
        maxWidth: "none",
        maxHeight: "none",
        margin: "0",
        transform: "none",
        opacity: "1",
        pointerEvents: "none",
      });
      ghost.style.transformOrigin = src.style.transformOrigin || "50% 50%";
      layer.appendChild(ghost);
      noteTicketLeaving();
      const d = ticketOut.duration as number;
      animate(ghost, { scale: [1, scaleTo] }, ticketOut);
      animate(ghost, { opacity: [1, 0] }, { duration: d, ease: [0.9, 0, 1, 1] }).then(() => ghost.remove());
    },
    [ref, scaleTo],
  );
}
