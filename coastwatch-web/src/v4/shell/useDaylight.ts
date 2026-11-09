"use client";

import { useEffect, useState } from "react";
import { DAYLIGHT_CHANGED, readDaylight, setDaylight } from "./daylight";

/** The daylight state, after hydration (the server never knows it), shared by every control that shows it. */
export function useDaylight(): [boolean, (on: boolean) => void] {
  const [day, setDay] = useState(false);
  useEffect(() => {
    const read = () => setDay(readDaylight());
    read();
    window.addEventListener(DAYLIGHT_CHANGED, read);
    return () => window.removeEventListener(DAYLIGHT_CHANGED, read);
  }, []);
  return [day, setDaylight];
}
