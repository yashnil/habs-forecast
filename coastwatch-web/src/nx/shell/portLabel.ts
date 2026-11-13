"use client";

import { useLayoutEffect, useSyncExternalStore } from "react";

/**
 * What the brand pill's port control says when the page is about something other than a port (v14 README §2): Research
 * says where its station is ("Near Santa Cruz"), as the synthesis does. A page declares it with <PortLabel text=… />;
 * without one the control names the current port. Choosing a port from it works as always.
 */
let label: string | null = null;
const listeners = new Set<() => void>();
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
};
export const usePortLabel = () => useSyncExternalStore(subscribe, () => label, () => null);

export function PortLabel({ text }: { text: string | null }) {
  useLayoutEffect(() => {
    label = text;
    listeners.forEach((l) => l());
    return () => {
      if (label === text) {
        label = null;
        listeners.forEach((l) => l());
      }
    };
  }, [text]);
  return null;
}
