// Brand icons drawn from Neritic's C2 mark (src/v4/brand/c2.ts, the same geometry the header uses):
// the favicon (.ico only: the hand-drawn 16 and 24 px masters plus 32/48 renders), the PWA icons
// and the Apple touch icon. No SVG favicon is declared: Chrome and Firefox prefer an SVG over the
// ICO, and the italic n only survives at 16 px in its hand-drawn master
// (docs/coastwatch/v4/brand/final/recommendation.md). Rerun after changing the mark:
//   node scripts/build_brand_assets.mjs
import { chromium } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { c2Svg } from "../src/v4/brand/c2.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const masters = join(root, "../docs/coastwatch/v4/brand/round2/icons");

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 1 });
async function png(svg, size) {
  await page.setViewportSize({ width: size, height: size });
  await page.setContent(`<html><body style="margin:0;background:transparent">${svg.replace(/width="\d+" height="\d+"/, `width="${size}" height="${size}"`)}</body></html>`);
  return page.screenshot({ omitBackground: true, clip: { x: 0, y: 0, width: size, height: size } });
}
const out = (p, b) => (mkdirSync(dirname(join(root, p)), { recursive: true }), writeFileSync(join(root, p), b));

// favicon.ico: PNG-compressed entries (supported by every current browser); 16 and 24 are drawn by hand
const entries = [
  [16, readFileSync(join(masters, "c2-16.png"))],
  [24, readFileSync(join(masters, "c2-24.png"))],
  [32, await png(c2Svg({ tone: "favicon" }), 32)],
  [48, await png(c2Svg({ tone: "favicon" }), 48)],
];
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
// app icons: the full-bleed square master (the OS rounds the corners)
out("src/app/apple-icon.png", await png(c2Svg({ square: true }), 180));
out("public/icons/icon-192.png", await png(c2Svg({ square: true }), 192));
out("public/icons/icon-512.png", await png(c2Svg({ square: true }), 512));
// maskable: the symbol inside the centre 80 % circle, so the contour stubs end inside the safe zone
out("public/icons/maskable-512.png", await png(c2Svg({ square: true, inset: 4.8 }), 512));
await browser.close();
console.log("brand assets written");
