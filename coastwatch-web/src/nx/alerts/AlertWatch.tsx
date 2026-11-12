"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { runCheck } from "./check";

/**
 * Runs the alert check when a page opens, and again when the reader comes back to the tab or the connection
 * returns. Mounted once in the shell; it draws nothing. The check itself decides whether one is due (at most
 * one every 30 minutes) and does nothing offline or when no port is followed, so this costs a page nothing.
 */
export function AlertWatch() {
  const pathname = usePathname();
  useEffect(() => {
    // after the page has drawn: a check never delays what the reader came for
    const t = window.setTimeout(() => void runCheck(), 1500);
    return () => window.clearTimeout(t);
  }, [pathname]);
  useEffect(() => {
    const again = () => {
      if (document.visibilityState === "visible") void runCheck();
    };
    document.addEventListener("visibilitychange", again);
    window.addEventListener("online", again);
    return () => {
      document.removeEventListener("visibilitychange", again);
      window.removeEventListener("online", again);
    };
  }, []);
  return null;
}
