"use client";

import { useCallback, useSyncExternalStore } from "react";
import { canHover } from "./reduced";

/** The shared store behind useFocusLink: which subject is lit, by id. */
let current: string | null = null;
const subs = new Set<() => void>();
export const focusLinkStore = {
  get: () => current,
  set(id: string | null) {
    if (id === current) return;
    current = id;
    subs.forEach((f) => f());
  },
  subscribe(f: () => void) {
    subs.add(f);
    return () => subs.delete(f);
  },
};

/**
 * Hovering or focusing a row lights its subject on the stage. `bind(id)` goes on the row; OceanStage reads
 * the store when it has no `focus` prop of its own. Pointer hover counts only on devices with a real hover;
 * keyboard focus always counts.
 *
 * Use for: a row whose subject is on the map (a fishery's notice line, a test's site). Not for: anything
 * without a place on the stage.
 */
export function useFocusLink() {
  const focus = useSyncExternalStore(focusLinkStore.subscribe, focusLinkStore.get, () => null);
  const setFocus = useCallback((id: string | null) => focusLinkStore.set(id), []);
  const bind = useCallback(
    (id: string) => ({
      onPointerEnter: (e: { pointerType?: string }) => {
        if (e.pointerType === "mouse" && canHover()) focusLinkStore.set(id);
      },
      onPointerLeave: () => {
        if (focusLinkStore.get() === id) focusLinkStore.set(null);
      },
      onFocus: () => focusLinkStore.set(id),
      onBlur: () => {
        if (focusLinkStore.get() === id) focusLinkStore.set(null);
      },
    }),
    [],
  );
  return { focus, setFocus, bind };
}
