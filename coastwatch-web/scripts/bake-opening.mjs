// Bakes the opening's canyon still (docs/coastwatch/v12/design-system/README.md §8, journeys.md Journey 1):
// the lit Monterey Canyon relief from pipeline/scripts/build_chart.py (NOAA CRM 3", seafloor x4, land x1),
// dithered once at build time with Paper Shaders' image-dithering (8x8 Bayer, two tones), and saved as an
// alpha mask so the page can ink it in any chart mode's colours (CSS mask, never a runtime shader).
//
// usage: node scripts/bake-opening.mjs [relief.png]   (default: <tmp>/neritic-chart/opening-relief.png)
// out:   public/chart/opening-canyon.webp (lossless, alpha = ink)
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import { chromium } from "playwright-core";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = process.argv[2] ?? join(tmpdir(), "neritic-chart", "opening-relief.png");
const out = join(root, "public", "chart", "opening-canyon.webp");
const shaders = join(root, "node_modules", "@paper-design", "shaders", "dist");

const relief = readFileSync(src);
const { width, height } = await sharp(relief).metadata();
// dither parameters: an 8x8 Bayer screen of 2 px cells, two tones (ink where the relief is in shadow)
const SIZE = 2;

const page = `<!doctype html><html><body style="margin:0;background:#000">
<div id="m" style="width:${width}px;height:${height}px"></div>
<script type="module">
import { ShaderMount, imageDitheringFragmentShader, getShaderColorFromString, DitheringTypes, ShaderFitOptions, defaultObjectSizing } from "/shaders/index.js";
const img = new Image();
img.src = "/relief.png";
await img.decode();
const s = defaultObjectSizing;
new ShaderMount(document.getElementById("m"), imageDitheringFragmentShader, {
  u_image: img,
  u_colorFront: getShaderColorFromString("#000000"),
  u_colorBack: getShaderColorFromString("#ffffff"),
  u_colorHighlight: getShaderColorFromString("#000000"),
  u_type: DitheringTypes["8x8"],
  u_pxSize: ${SIZE},
  u_colorSteps: 2,
  u_originalColors: false,
  u_inverted: true,
  u_fit: ShaderFitOptions.cover,
  u_rotation: s.rotation, u_scale: s.scale, u_offsetX: s.offsetX, u_offsetY: s.offsetY,
  u_originX: s.originX, u_originY: s.originY, u_worldWidth: s.worldWidth, u_worldHeight: s.worldHeight,
}, { preserveDrawingBuffer: true }, 0, 0, 1, 1e8);
requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(() => (window.__done = true))));
</script></body></html>`;

const browser = await chromium.launch({ channel: process.env.CI ? undefined : "chrome", args: ["--use-angle=metal", "--enable-gpu"] });
const ctx = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1 });
const tab = await ctx.newPage();
await tab.route("http://bake.local/**", async (r) => {
  const p = new URL(r.request().url()).pathname;
  if (p === "/") return r.fulfill({ contentType: "text/html", body: page });
  if (p === "/relief.png") return r.fulfill({ contentType: "image/png", body: relief });
  if (p.startsWith("/shaders/")) return r.fulfill({ contentType: "text/javascript", body: readFileSync(join(shaders, p.slice("/shaders/".length))) });
  return r.fulfill({ status: 404, body: "" });
});
await tab.goto("http://bake.local/");
await tab.waitForFunction(() => window.__done === true, null, { timeout: 30_000 });
const shot = await tab.locator("canvas").screenshot({ type: "png" });
await browser.close();

// ink = dark pixels of the dither; alpha carries it, the colour comes from the page
const lum = await sharp(shot).resize(width, height, { fit: "fill" }).greyscale().raw().toBuffer();
const rgba = Buffer.alloc(width * height * 4);
for (let i = 0; i < width * height; i++) rgba[i * 4 + 3] = 255 - lum[i];
await sharp(rgba, { raw: { width, height, channels: 4 } }).webp({ lossless: true, effort: 6 }).toFile(out);
const bytes = readFileSync(out).length;
console.log(`opening still -> ${out} (${width}x${height}, ${(bytes / 1024).toFixed(0)} KB)`);
