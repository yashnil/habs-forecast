import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { MARK_FIELD, MARK_INDEX_DEPTH } from "@/v4/brand/mark.geometry";
import { ContourLoader, ISOBATH_200 } from "@/v4/ui/ContourLoader";
import { T } from "@/v4/motion";

// v9 motion (docs/coastwatch/v9/motion.md §2, §4): one token set, no overshoot except under a
// finger, every raster cross-fades, flights are capped, and the loader draws a real isobath
const src = (p: string) => readFileSync(path.resolve(__dirname, "../../src", p), "utf8");
const theme = src("v4/theme.css");
const globals = src("app/globals.css");
const tokenOf = (name: string) => theme.match(new RegExp(`--${name}:\\s*([^;]+);`))?.[1].trim();

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = path.join(dir, f);
    return statSync(p).isDirectory() ? files(p) : /\.(tsx?|css)$/.test(f) ? [p] : [];
  });
}

describe("motion tokens", () => {
  it("has the v9 values, and the CSS and script durations agree", () => {
    expect(tokenOf("nx-t-press")).toBe("90ms");
    expect(tokenOf("nx-t-release")).toBe("220ms");
    expect(tokenOf("nx-t-layer")).toBe("220ms");
    expect(tokenOf("nx-t-page")).toBe("200ms");
    expect(T.press).toBe(0.09);
    expect(T.layer).toBe(0.22);
    expect(T.page).toBe(0.2);
  });

  it("keeps no parallel or alias motion tokens", () => {
    const all = files(path.resolve(__dirname, "../../src")).map((f) => readFileSync(f, "utf8")).join("\n");
    for (const gone of ["--cw-ease-settle", "--nx-ease-in-out", "--nx-t-quick", "--nx-t-base"]) expect(all).not.toContain(gone);
  });

  it("the CSS spring never overshoots by more than 3 %, and controls do not use it", () => {
    const curve = theme.match(/--nx-ease-spring:\s*linear\(([^)]+)\)/)?.[1];
    expect(curve).toBeTruthy();
    const values = curve!.split(",").map((x) => parseFloat(x.trim()));
    expect(Math.max(...values)).toBeLessThanOrEqual(1.03);
    // press, release, focus ring and lift ease out; only the sheet's drag release springs
    const controls = theme.slice(theme.indexOf("/* ---- interaction: one feel"), theme.indexOf("/* the deck: its edge"));
    expect(controls).not.toContain("--nx-ease-spring");
    expect(controls).not.toMatch(/\b90ms\b/);
    expect(globals).not.toContain("--nx-ease-spring");
    expect(src("components/MobileSheet.tsx")).toMatch(/via === "drag" \? "transform var\(--nx-t-sheet\) var\(--nx-ease-spring\)"/);
  });

  it("panels slide 12 px, in CSS and in the Explore panel", () => {
    expect(theme).toMatch(/@keyframes nx-panel \{\s*from \{ opacity: 0; transform: translateX\(calc\(12px/);
    expect(src("v4/explore/Explore.tsx")).toContain("initial={{ opacity: 0, x: 12 }}");
  });
});

describe("layer cross-fade", () => {
  it("every data raster, the satellite observation included, fades 220 ms on a layer switch and cuts on a new day (0 under reduced motion)", () => {
    const map = src("components/map/MapCanvas.tsx");
    expect(map).toContain("export const RASTER_FADE_MS = 220;");
    expect(map).toMatch(/const RASTER_PAINT = \{[^}]*"raster-fade-duration": RASTER_FADE_MS/);
    expect(map).toMatch(/prefers-reduced-motion: reduce\)"\)\.matches \? 0 : RASTER_FADE_MS/);
    const rasters = [...map.matchAll(/<Layer id="([\w-]+-raster)"[^>]*paint=\{([^}]+\}?)/g)];
    expect(rasters.map((r) => r[1])).toEqual(expect.arrayContaining(["satellite-raster", "observation-raster", "forecast-raster", "sst-raster", "outlook-raster", "estimate-raster", "satellite-age-raster"]));
    for (const [, id, paint] of rasters) expect(paint, id).toMatch(/paintFor\("|"raster-fade-duration": fadeFor\("/);
    // a source replacing the same layer (a new day) cuts: its old day is already gone, a fade would dip to the bare sea
    expect(map).toContain("shownLayers.current.has(layer) ? 0 : fadeMs");
  });
});

describe("journeys", () => {
  async function load(renderer: string | null) {
    vi.resetModules();
    vi.stubGlobal("window", { matchMedia: () => ({ matches: false }) });
    vi.stubGlobal("sessionStorage", { getItem: () => null, setItem: () => {} });
    const gl = renderer == null ? null : { getExtension: (n: string) => (n === "WEBGL_debug_renderer_info" ? { UNMASKED_RENDERER_WEBGL: 1 } : null), getParameter: () => renderer };
    vi.stubGlobal("document", { createElement: () => ({ getContext: () => gl }) });
    vi.stubGlobal("requestAnimationFrame", () => 0);
    vi.stubGlobal("cancelAnimationFrame", () => {});
    return import("@/lib/camera");
  }
  const fake = (centre: { lng: number; lat: number }) => {
    const calls: { fn: string; opts: Record<string, unknown> }[] = [];
    const m = {
      cameraForBounds: () => ({ center: { lng: -124.2, lat: 41.75 }, zoom: 9 }),
      getCenter: () => centre,
      getZoom: () => 6,
      once: () => {},
      jumpTo: (opts: Record<string, unknown>) => calls.push({ fn: "jumpTo", opts }),
      easeTo: (opts: Record<string, unknown>) => calls.push({ fn: "easeTo", opts }),
      flyTo: (opts: Record<string, unknown>) => calls.push({ fn: "flyTo", opts }),
    };
    return { m, calls };
  };

  it("a long flight is capped at 1.8 s; a near pick eases in 600 ms", async () => {
    const { journeyTo, flightMs, FLIGHT_MAX_MS } = await load("ANGLE (Apple M2)");
    // San Diego to Crescent City, about 1100 km
    const far = fake({ lng: -117.2, lat: 32.7 });
    journeyTo(far.m as never, { center: [-124.2, 41.75], zoom: 9 });
    expect(far.calls[0].fn).toBe("flyTo");
    expect(far.calls[0].opts.duration).toBe(1800);
    expect(FLIGHT_MAX_MS).toBe(1800);
    expect(flightMs(100)).toBe(1000);
    const near = fake({ lng: -124.1, lat: 41.6 });
    journeyTo(near.m as never, { center: [-124.2, 41.75], zoom: 9 });
    expect(near.calls[0]).toMatchObject({ fn: "easeTo", opts: { duration: 600 } });
  });

  it("a cluster click is a near journey, not its own timing", () => {
    const map = src("components/map/MapCanvas.tsx");
    expect(map).not.toMatch(/duration: 450\b/);
    expect(map).toMatch(/tier === "short" \? SHORT_MS : NEAR_MS/);
  });
});

describe("contour loader", () => {
  it("draws Monterey Canyon's 200 m isobath from the brand's field, unchanged", () => {
    const real = MARK_FIELD.lines.filter((l) => l.depth === MARK_INDEX_DEPTH).map((l) => l.d);
    expect(real.length).toBeGreaterThan(0);
    expect([...ISOBATH_200]).toEqual(real);
    const svg = renderToStaticMarkup(createElement(ContourLoader));
    for (const d of real) expect(svg).toContain(`d="${d}"`);
  });

  it("sounds only while loading; still otherwise (and the CSS stills it under reduced motion)", () => {
    const loading = renderToStaticMarkup(createElement(ContourLoader, { loading: true }));
    const still = renderToStaticMarkup(createElement(ContourLoader, { loading: false }));
    expect(loading).toContain('class="nx-sound"');
    expect(loading).toContain('data-state="loading"');
    expect(still).not.toContain("nx-sound");
    expect(still).toContain('data-state="still"');
    expect(theme).toMatch(/@media \(prefers-reduced-motion: reduce\) \{\s*\.nx-sound \{ animation: none; \}/);
  });

  it("no procedural curves stand in for bathymetry", () => {
    for (const f of files(path.resolve(__dirname, "../../src"))) expect(readFileSync(f, "utf8"), f).not.toMatch(/M-20 \$\{|nx-sweep"/);
  });
});
