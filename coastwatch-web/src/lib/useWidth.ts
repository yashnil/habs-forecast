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
