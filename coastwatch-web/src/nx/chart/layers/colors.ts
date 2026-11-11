"use client";

/**
 * The active chart mode's colours for the GPU, read from the tokens on <html> (src/nx/ds/tokens.css: day,
 * dusk and night, each specified). No colour is written here: a token's value, in whatever CSS form the
 * token file uses, is resolved by the browser itself through a 1x1 canvas and handed over as linear 0..1 RGBA.
 */
export type RGBA = [number, number, number, number];

let ctx: CanvasRenderingContext2D | null = null;

/** A CSS colour as RGBA 0..1 (straight alpha). Unparseable values read as transparent. */
export function cssColour(value: string): RGBA {
  if (!value) return [0, 0, 0, 0];
  if (!ctx) {
    const c = document.createElement("canvas");
    c.width = c.height = 1;
    ctx = c.getContext("2d", { willReadFrequently: true });
  }
  if (!ctx) return [0, 0, 0, 0];
  ctx.clearRect(0, 0, 1, 1);
  ctx.fillStyle = "transparent";
  ctx.fillStyle = value;
  ctx.fillRect(0, 0, 1, 1);
  const d = ctx.getImageData(0, 0, 1, 1).data;
  const a = d[3] / 255;
  // getImageData un-premultiplies; keep straight alpha
  return [d[0] / 255, d[1] / 255, d[2] / 255, a];
}

/** A token of the current mode (`--chl-3`, `--ink`). */
export function token(name: string, el: Element = document.documentElement): RGBA {
  return cssColour(getComputedStyle(el).getPropertyValue(name).trim());
}

export type ModeColours = {
  chl: RGBA[];
  ink: RGBA;
  ink2: RGBA;
  hatch: RGBA;
  paper: RGBA;
  halo: RGBA;
  streak: RGBA;
  isobath: RGBA;
};

/** Everything the data layers draw with, in the current mode. */
export function modeColours(el: Element = document.documentElement): ModeColours {
  const cs = getComputedStyle(el);
  const v = (k: string) => cssColour(cs.getPropertyValue(k).trim());
  return {
    chl: Array.from({ length: 8 }, (_, i) => v(`--chl-${i}`)),
    ink: v("--ink"),
    ink2: v("--ink-2"),
    hatch: v("--hatch"),
    paper: v("--paper"),
    halo: v("--halo"),
    streak: v("--streak"),
    isobath: v("--isobath"),
  };
}

/** Calls `fn` whenever the chart mode changes (the shell sets data-chart-mode on <html>). */
export function onModeChange(fn: () => void): () => void {
  const mo = new MutationObserver(fn);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-chart-mode"] });
  return () => mo.disconnect();
}
