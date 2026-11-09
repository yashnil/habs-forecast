import { readFileSync } from "node:fs";
import path from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Fold, Reveal, Segmented, Sheet } from "@/v4/motion";
import { belowFold } from "@/v4/motion/Reveal";
import { DISMISS_SHARE, shouldDismiss } from "@/v4/motion/Sheet";
import { glideFrom } from "@/v4/motion/Segmented";
import { LATE_DELAY_MS, LATE_MIN_MS, lateTimer } from "@/v4/motion/Skeleton";
import { focusLinkStore } from "@/v4/motion/focusLink";
import { PLAYBACK_MS, playbackClock, stopsPlayback } from "@/v4/motion/playback";
import { isRouteChange, mayTransition, shouldIntercept } from "@/v4/motion/routeFilter";

const el = (C: unknown, props: object, ...kids: unknown[]) => createElement(C as never, props as never, ...(kids as never[]));
const src = (p: string) => readFileSync(path.resolve(__dirname, "../../src", p), "utf8");
const css = src("v4/motion/motion.css");

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Reveal", () => {
  it("hides only what is below the fold at hydration", () => {
    expect(belowFold(900, 900)).toBe(true);
    expect(belowFold(1200, 900)).toBe(true);
    expect(belowFold(400, 900)).toBe(false);
    expect(belowFold(-50, 900)).toBe(false);
  });
  it("is plain visible markup in the server HTML", () => {
    const html = renderToStaticMarkup(el(Reveal, {}, createElement("h2", null, "A chapter")));
    expect(html).toContain("A chapter");
    expect(html).not.toMatch(/opacity|hidden|data-reveal=/);
  });
  it("group staggers 50 ms apart, at most five, and reduced motion keeps only the fade", () => {
    expect(css).toMatch(/:nth-child\(2\) \{ animation-delay: 50ms/);
    expect(css).toMatch(/:nth-child\(n \+ 5\) \{ animation-delay: 200ms/);
    expect(css).toMatch(/@media \(prefers-reduced-motion: reduce\) \{[\s\S]*animation-name: nx-m-fade/);
    expect(css).toMatch(/@keyframes nx-m-reveal \{\s*from \{ opacity: 0; transform: translateY\(calc\(12px \* var\(--nx-shift, 1\)\)\)/);
  });
});

describe("usePlayback's clock", () => {
  it("only ever lands on whole frames, and stops at the end", () => {
    vi.useFakeTimers();
    const seen: number[] = [];
    let stopped = false;
    const c = playbackClock({ count: 3, intervalMs: PLAYBACK_MS, loop: false, onStep: (i) => seen.push(i), onStop: () => (stopped = true) });
    c.start(0);
    expect(c.running()).toBe(true);
    vi.advanceTimersByTime(PLAYBACK_MS / 2);
    expect(seen).toEqual([]); // nothing between frames
    vi.advanceTimersByTime(PLAYBACK_MS * 5);
    expect(seen).toEqual([1, 2]);
    expect(stopped).toBe(true);
    expect(seen.every(Number.isInteger)).toBe(true);
  });
  it("loops when asked, and a single frame never plays", () => {
    vi.useFakeTimers();
    const seen: number[] = [];
    const c = playbackClock({ count: 2, intervalMs: 100, loop: true, onStep: (i) => seen.push(i), onStop: () => {} });
    c.start(1);
    vi.advanceTimersByTime(250);
    expect(seen).toEqual([0, 1]);
    c.stop();
    let stops = 0;
    const one = playbackClock({ count: 1, intervalMs: 100, loop: true, onStep: () => seen.push(99), onStop: () => stops++ });
    one.start(0);
    vi.advanceTimersByTime(500);
    expect(seen).not.toContain(99);
    expect(stops).toBe(1);
  });
  it("any input stops it, except on the Play/Pause control itself", () => {
    expect(stopsPlayback({ closest: () => null } as unknown as EventTarget)).toBe(true);
    expect(stopsPlayback({ closest: () => ({}) } as unknown as EventTarget)).toBe(false);
    expect(stopsPlayback(null)).toBe(true);
    // Space in the time rail is its own play/pause key: the "any input" rule must not pause first and let it restart
    const inRail = (radio: boolean) => ({ closest: (s: string) => (s === "[data-playback-space]" ? {} : s.startsWith("button") && !radio ? {} : null) }) as unknown as EventTarget;
    expect(stopsPlayback(inRail(true), " ")).toBe(false);
    expect(stopsPlayback(inRail(true), "ArrowRight")).toBe(true);
    expect(stopsPlayback(inRail(false), " ")).toBe(true);
  });
});

describe("route interceptor's filter", () => {
  const loc = { origin: "https://neritic.app", pathname: "/today" };
  const click = { button: 0, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false };
  const link = (href: string, more = {}) => ({ href, ...more });
  it("takes a plain click to another page of the app", () => {
    expect(shouldIntercept(click, link("/coast"), loc)).toBe(true);
    expect(shouldIntercept(click, link("/?view=map"), loc)).toBe(true);
    expect(shouldIntercept(click, link("/research?station=x"), loc)).toBe(true);
  });
  it("leaves modifier clicks, other buttons, targets, downloads and marked links alone", () => {
    for (const k of ["metaKey", "ctrlKey", "shiftKey", "altKey"]) expect(shouldIntercept({ ...click, [k]: true }, link("/coast"), loc)).toBe(false);
    expect(shouldIntercept({ ...click, button: 1 }, link("/coast"), loc)).toBe(false);
    expect(shouldIntercept({ ...click, defaultPrevented: true }, link("/coast"), loc)).toBe(false);
    expect(shouldIntercept(click, link("/coast", { target: "_blank" }), loc)).toBe(false);
    expect(shouldIntercept(click, link("/coast", { target: "_self" }), loc)).toBe(true);
    expect(shouldIntercept(click, link("/coast", { download: true }), loc)).toBe(false);
    expect(shouldIntercept(click, link("/coast", { noTransition: true }), loc)).toBe(false);
  });
  it("leaves the same page, hash-only and query-only changes, other sites and files alone", () => {
    expect(isRouteChange("/today", loc)).toBe(false);
    expect(isRouteChange("#sea", loc)).toBe(false);
    expect(isRouteChange("/today?port=592", loc)).toBe(false);
    expect(isRouteChange("https://cdfw.ca.gov/x", loc)).toBe(false);
    expect(isRouteChange("mailto:a@b.c", loc)).toBe(false);
    expect(isRouteChange("/api/briefing", loc)).toBe(false);
    expect(isRouteChange("/data/v1/manifest.json", loc)).toBe(false);
    expect(isRouteChange("/field-pack.pdf", loc)).toBe(false);
    expect(isRouteChange("/coast", loc)).toBe(true);
  });
  it("never starts one without the API, under reduced motion, during the splash, offline, hidden or on a saved copy", () => {
    const ok = { hasApi: true, reduced: false, splash: false, hidden: false, online: true, saved: false };
    expect(mayTransition(ok)).toBe(true);
    for (const k of ["hasApi", "online"] as const) expect(mayTransition({ ...ok, [k]: false })).toBe(false);
    for (const k of ["reduced", "splash", "hidden", "saved"] as const) expect(mayTransition({ ...ok, [k]: true })).toBe(false);
  });
});

describe("Sheet", () => {
  it("a drag dismisses past a share of its height or on a flick", () => {
    expect(shouldDismiss(100, 800, 0.1)).toBe(false);
    expect(shouldDismiss(800 * DISMISS_SHARE + 1, 800, 0)).toBe(true);
    expect(shouldDismiss(40, 800, 0.8)).toBe(true);
    expect(shouldDismiss(3, 800, 2)).toBe(false); // a tap with a twitch is not a flick
  });
  it("renders nothing closed, and a labelled modal dialog open", () => {
    expect(renderToStaticMarkup(el(Sheet, { open: false, onClose: () => {}, title: "T" }, "body"))).toBe("");
    const html = renderToStaticMarkup(el(Sheet, { open: true, onClose: () => {}, title: "Dungeness" }, "body"));
    expect(html).toMatch(/<dialog/);
    expect(html).toMatch(/aria-modal="true"/);
    expect(html).toMatch(/aria-labelledby/);
    expect(html).toContain("Dungeness");
  });
  it("travels 24 px on desktop and from the bottom on phones", () => {
    expect(css).toMatch(/translateX\(calc\(24px \* var\(--nx-shift, 1\)\)\)/);
    expect(css).toMatch(/width: 420px/);
    expect(css).toMatch(/--nx-m-off: translateY\(calc\(100% \* var\(--nx-shift, 1\)\)\)/);
  });
});

describe("Fold and Segmented", () => {
  it("Fold keeps its words in the DOM while folded, and marks them inert", () => {
    const html = renderToStaticMarkup(el(Fold, { title: "All periods" }, "Tonight, north wind"));
    expect(html).toContain("Tonight, north wind");
    expect(html).toContain('aria-expanded="false"');
    expect(html).toMatch(/inert/);
    expect(html).toContain('class="nx-fold"');
  });
  it("Segmented has the thumb in the active item (right without script) and the right roles", () => {
    const html = renderToStaticMarkup(createElement(Segmented, { value: "b", onChange: () => {}, label: "Window", options: [{ value: "a", label: "A" }, { value: "b", label: "B" }] }));
    expect(html).toContain('role="radiogroup"');
    expect(html.match(/data-glide-thumb/g)).toHaveLength(1);
    expect(html.indexOf('data-seg="b"')).toBeLessThan(html.indexOf("data-glide-thumb"));
    expect(html).toContain('aria-checked="true"');
  });
  it("a quick second choice glides from where the thumb is, not from where it was going", () => {
    const moving = { x: 100, w: 50, fromX: 0, fromW: 40, at: 1000 };
    const halfway = glideFrom(moving, 1130, 260)!;
    expect(halfway.x).toBeGreaterThan(0);
    expect(halfway.x).toBeLessThan(100);
    expect(glideFrom(moving, 2000, 260)).toEqual({ x: 100, w: 50 });
    expect(glideFrom(null, 0, 260)).toBeNull();
  });
});

describe("Loading (useLate's timer)", () => {
  it("shows nothing for 300 ms, and a shown skeleton stays at least 300 ms", () => {
    vi.useFakeTimers();
    let t = 0;
    const now = () => t;
    const events: boolean[] = [];
    const late = lateTimer({ delay: LATE_DELAY_MS, min: LATE_MIN_MS, onChange: (s) => events.push(s) });
    late.set(true, now);
    vi.advanceTimersByTime(250);
    late.set(false, now);
    vi.advanceTimersByTime(1000);
    expect(events).toEqual([]); // a fast load never shows it
    t = 1000;
    late.set(true, now);
    t = 1300;
    vi.advanceTimersByTime(300);
    expect(events).toEqual([true]);
    t = 1350;
    late.set(false, now);
    vi.advanceTimersByTime(200);
    expect(events).toEqual([true]);
    vi.advanceTimersByTime(60);
    expect(events).toEqual([true, false]);
  });
});

describe("focus link", () => {
  it("publishes one id at a time to its subscribers", () => {
    const seen: (string | null)[] = [];
    const off = focusLinkStore.subscribe(() => seen.push(focusLinkStore.get()));
    focusLinkStore.set("a");
    focusLinkStore.set("a");
    focusLinkStore.set(null);
    off();
    focusLinkStore.set("b");
    expect(seen).toEqual(["a", null]);
    focusLinkStore.set(null);
  });
});

describe("motion.css", () => {
  it("animates no layout and no shimmer, and names the route transition's elements", () => {
    expect(css).not.toMatch(/transition:\s*all/);
    expect(css).not.toMatch(/shimmer/i);
    expect(css).toMatch(/\.nx-topbar \{ view-transition-name: nx-topbar; \}/);
    expect(css).toMatch(/\.nx-vt-stage \{ view-transition-name: nx-stage; \}/);
    expect(css).toMatch(/::view-transition \{ pointer-events: none; \}/);
    expect(css).toMatch(/@supports \(animation-timeline: scroll\(\)\)/);
    expect(css).toMatch(/opacity: 0\.55; transform: scale\(1\.04\)/);
    expect(src("app/globals.css").match(/motion\/motion\.css/g)).toHaveLength(1);
  });
});
