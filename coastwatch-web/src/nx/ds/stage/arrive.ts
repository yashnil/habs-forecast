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
  const last = useRef<{ node: HTMLElement; rect: DOMRect; origin: string } | null>(null);
  // keep a fresh copy's source: measured at each commit while mounted
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    last.current = { node: el, rect: el.getBoundingClientRect(), origin: el.style.transformOrigin };
  });
  useLayoutEffect(
    () => () => {
      const src = last.current?.node;
      const layer = document.getElementById(GHOST_LAYER_ID);
      if (!src || !layer || isPhoneNow() || !src.isConnected) return;
      const r = src.getBoundingClientRect();
      if (!r.width || !r.height) return;
      const ghost = src.cloneNode(true) as HTMLElement;
      ghost.removeAttribute("id");
      ghost.querySelectorAll("[id]").forEach((n) => n.removeAttribute("id"));
      ghost.setAttribute("aria-hidden", "true");
      ghost.setAttribute("inert", "");
      ghost.removeAttribute("data-testid");
      ghost.querySelectorAll("[data-testid]").forEach((n) => n.removeAttribute("data-testid"));
      // a leaving copy is decoration: it must not answer for the page's header or its ids
      ghost.removeAttribute("data-nx-desk-header");
      ghost.querySelectorAll("[data-nx-desk-header], [id]").forEach((n) => { n.removeAttribute("data-nx-desk-header"); n.removeAttribute("id"); });
      ghost.setAttribute("aria-hidden", "true");
      ghost.setAttribute("inert", "");
      const lr = layer.getBoundingClientRect();
      Object.assign(ghost.style, { position: "absolute", left: `${r.left - lr.left}px`, top: `${r.top - lr.top}px`, right: "auto", bottom: "auto", width: `${r.width}px`, height: `${r.height}px`, margin: "0", transform: "none", opacity: "1", pointerEvents: "none" });
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
