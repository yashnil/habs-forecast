// Brand assets drawn from Neritic's contour mark (src/v4/brand/mark.ts, the same geometry the header
// uses; the geometry itself comes from real bathymetry via scripts/brand/build_mark.py):
// - src/app/favicon.ico: 16 px (the two-line micro master), 24/32/48 px (the three-line small master)
// - src/app/apple-icon.png, public/icons/icon-{192,512}.png: the full master, full bleed (the OS rounds it)
// - public/icons/maskable-512.png: the full master inside the maskable safe zone
// - public/icons/brand/*.svg: the masters (light, dark, mono) and the lockups, for anyone who needs the files
// No SVG favicon is declared: browsers prefer it over the ICO, and the ICO's 16 px entry is the
// optical size drawn for a tab strip. Rerun after changing the mark:
//   node scripts/build_brand_assets.mjs
import { chromium } from "@playwright/test";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

// load the TypeScript brand module without a loader: transpile it and its geometry to a temp dir
const tmp = mkdtempSync(join(tmpdir(), "nx-brand-"));
for (const name of ["mark", "mark.geometry"]) {
  const src = readFileSync(join(root, `src/v4/brand/${name}.ts`), "utf8");
  const js = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
  writeFileSync(join(tmp, `${name}.mjs`), js.replace(/from "\.\/mark\.geometry"/g, 'from "./mark.geometry.mjs"'));
}
const { markSvg, lockupSvg } = await import(pathToFileURL(join(tmp, "mark.mjs")).href);

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 1 });
async function png(svg, size) {
  await page.setViewportSize({ width: size, height: size });
  await page.setContent(`<html><body style="margin:0;background:transparent">${svg.replace(/width="[\d.]+" height="[\d.]+"/, `width="${size}" height="${size}"`)}</body></html>`);
  return page.screenshot({ omitBackground: true, clip: { x: 0, y: 0, width: size, height: size } });
}
const out = (p, b) => (mkdirSync(dirname(join(root, p)), { recursive: true }), writeFileSync(join(root, p), b));

// favicon.ico: PNG-compressed entries (supported by every current browser)
const entries = [];
for (const s of [16, 24, 32, 48]) entries.push([s, await png(markSvg({ size: s }), s)]);
const head = Buffer.alloc(6 + 16 * entries.length);
head.writeUInt16LE(0, 0); head.writeUInt16LE(1, 2); head.writeUInt16LE(entries.length, 4);
let offset = head.length;
entries.forEach(([s, b], i) => {
  const e = 6 + 16 * i;
  head.writeUInt8(s, e); head.writeUInt8(s, e + 1); head.writeUInt8(0, e + 2); head.writeUInt8(0, e + 3);
  head.writeUInt16LE(1, e + 4); head.writeUInt16LE(32, e + 6);
  head.writeUInt32LE(b.length, e + 8); head.writeUInt32LE(offset, e + 12);
  offset += b.length;
});
out("src/app/favicon.ico", Buffer.concat([head, ...entries.map(([, b]) => b)]));
// app icons: the full master, full bleed
out("src/app/apple-icon.png", await png(markSvg({ square: true, tier: "full" }), 180));
out("public/icons/icon-192.png", await png(markSvg({ square: true, tier: "full" }), 192));
out("public/icons/icon-512.png", await png(markSvg({ square: true, tier: "full" }), 512));
// maskable: the drawing inside the centre 80 % circle (Android crops to a circle or squircle)
out("public/icons/maskable-512.png", await png(markSvg({ square: true, inset: 4.8, tier: "full" }), 512));
// the masters as files
out("public/icons/brand/mark-light.svg", markSvg({ tone: "light", size: 512, tier: "full" }));
out("public/icons/brand/mark-dark.svg", markSvg({ tone: "dark", size: 512, tier: "full" }));
out("public/icons/brand/mark-mono.svg", markSvg({ size: 512, tier: "full", mono: "#0b1820" }));
out("public/icons/brand/mark-small.svg", markSvg({ size: 48, tier: "small" }));
out("public/icons/brand/mark-micro.svg", markSvg({ size: 16, tier: "micro" }));
out("public/icons/brand/lockup-light.svg", lockupSvg({ tone: "light", height: 96 }));
out("public/icons/brand/lockup-dark.svg", lockupSvg({ tone: "dark", height: 96 }));
out("public/icons/brand/lockup-mono.svg", lockupSvg({ height: 96, mono: "#0b1820" }));
await browser.close();
console.log("brand assets written");
