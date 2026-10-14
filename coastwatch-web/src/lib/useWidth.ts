import { useEffect, useRef, useState } from "react";

/** The rendered width of an element (at least `min`), for charts drawn in pixels. */
export function useWidth<T extends HTMLElement>(initial = 720, min = 280): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(min, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, [min]);
  return [ref, w];
}

/** Whether the screen matches a width query (false until known on the client): pickers are popovers when wide, bottom sheets on phones. */
export function useWide(query = "(min-width: 640px)"): boolean {
  const [wide, setWide] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setWide(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return wide;
}
