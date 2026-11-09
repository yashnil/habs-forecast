"use client";

import { useEffect, useLayoutEffect, useMemo } from "react";
import { usePathname, useRouter } from "next/navigation";
import { isRouteChange, mayTransition, shouldIntercept } from "./routeFilter";
import { prefersReducedMotion } from "./reduced";

/** The longest the old page stays frozen waiting for the new one; past this the transition is skipped and the page keeps working. */
const ROUTE_WAIT_MS = 900;
/** The browser should ask for the update within a frame or two; if it has not by this, navigate regardless. */
const CALLBACK_GUARD_MS = 300;

type VT = { ready: Promise<void>; finished: Promise<void>; updateCallbackDone: Promise<void>; skipTransition: () => void };
type StartVT = (cb: () => Promise<void>) => VT;

let settleNow: (() => void) | null = null;

function canTransition(): boolean {
  const d = document as Document & { startViewTransition?: StartVT };
  return mayTransition({
    hasApi: typeof d.startViewTransition === "function",
    reduced: prefersReducedMotion(),
    splash: d.documentElement.classList.contains("nx-splash-on"),
    hidden: d.visibilityState !== "visible",
    online: navigator.onLine !== false,
    saved: !!(window as unknown as { __NERITIC_SAVED__?: unknown }).__NERITIC_SAVED__,
  });
}

/**
 * Runs `navigate` inside a view transition: the page freezes for a frame (the old snapshot), `navigate` runs,
 * and the callback resolves when the new route has committed (`routeRendered`, from RouteTransitions). Every
 * failure path still navigates: no API, a throw, a skip, a callback the browser never calls, a route that
 * takes too long. `navigate` runs at most once.
 */
export function withRouteTransition(navigate: () => boolean | void): void {
  if (!canTransition()) return void navigate();
  let ran = false;
  const run = () => {
    if (ran) return true;
    ran = true;
    return navigate();
  };
  let transition: VT | null = null;
  try {
    transition = (document as unknown as { startViewTransition: StartVT }).startViewTransition(
      () =>
        new Promise<void>((resolve) => {
          settleNow?.();
          let wait: ReturnType<typeof setTimeout> | null = null;
          const done = () => {
            if (wait != null) clearTimeout(wait);
            if (settleNow === done) settleNow = null;
            resolve();
          };
          settleNow = done;
          wait = setTimeout(() => {
            transition?.skipTransition();
            done();
          }, ROUTE_WAIT_MS);
          // false: the navigation was left to the browser (a full page load); nothing to wait for
          if (run() === false) done();
        }),
    );
  } catch {
    run();
    return;
  }
  const ignore = () => {};
  transition.ready.catch(ignore);
  transition.finished.catch(ignore);
  transition.updateCallbackDone.catch(ignore);
  setTimeout(() => {
    if (ran) return;
    transition?.skipTransition();
    run();
  }, CALLBACK_GUARD_MS);
}

/**
 * Route transitions for every link (mounted once, in MotionRoot). A trusted click on an in-app link is held,
 * and replayed on the same element inside the view transition, so next/link keeps all of its own behaviour
 * (onClick, scroll, replace, prefetch) and this file never has to know about it. Add `data-no-transition` to a
 * link that should cut.
 */
export function RouteTransitions() {
  const pathname = usePathname();
  useLayoutEffect(() => {
    // this commit carries the new route; the snapshot is taken at the next frame
    settleNow?.();
  }, [pathname]);
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (!e.isTrusted) return;
      const a = (e.target as Element | null)?.closest?.("a");
      if (!(a instanceof HTMLAnchorElement)) return;
      const loc = { origin: location.origin, pathname: location.pathname };
      if (!shouldIntercept(e, { href: a.getAttribute("href") ?? null, target: a.getAttribute("target"), download: a.hasAttribute("download"), noTransition: a.hasAttribute("data-no-transition") }, loc)) return;
      if (!canTransition()) return;
      e.preventDefault();
      e.stopPropagation();
      withRouteTransition(() => {
        const replay = new MouseEvent("click", { bubbles: true, cancelable: true, view: window });
        // an element gone by now cannot navigate; hard navigation is the honest fallback
        if (!a.isConnected) {
          location.assign(a.href);
          return false;
        }
        a.dispatchEvent(replay);
        // nothing handled it: the browser follows the link, a full load, and there is no new route to wait for
        return replay.defaultPrevented;
      });
    };
    document.addEventListener("click", onClick, true);
    return () => document.removeEventListener("click", onClick, true);
  }, []);
  return null;
}

/**
 * The app router with route transitions on `push` and `replace`, for programmatic navigations. Same-page
 * changes (a query, a hash) are not wrapped. Everything else is `useRouter()` unchanged.
 */
export function useTransitionRouter() {
  const router = useRouter();
  return useMemo(() => {
    const via = (method: "push" | "replace") => (href: string, opts?: Parameters<typeof router.push>[1]) => {
      const go = () => router[method](href, opts);
      if (isRouteChange(href, { origin: location.origin, pathname: location.pathname })) withRouteTransition(go);
      else go();
    };
    return { ...router, push: via("push"), replace: via("replace") };
  }, [router]);
}
