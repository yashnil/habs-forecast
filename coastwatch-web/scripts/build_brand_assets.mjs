// Brand icons drawn from the Neritic Signal mark (src/components/brand/BrandMark.tsx): the
// favicon (16/32/48 px PNGs in one .ico), the PWA icons and the Apple touch icon. Rerun after
// changing the mark: node scripts/build_brand_assets.mjs
import { chromium } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// the same drawing as BrandMark (32 x 32 viewBox); `pad`: safe-zone inset for maskable icons
export const markSvg = (pad = 0) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${-pad} ${-pad} ${32 + 2 * pad} ${32 + 2 * pad}">
${pad ? `<rect x="${-pad}" y="${-pad}" width="${32 + 2 * pad}" height="${32 + 2 * pad}" fill="#10263e"/>` : ""}<rect width="32" height="32" rx="${pad ? 0 : 8}" fill="#10263e"/>
<path d="M5 10.5c2.6-1.7 5.2-1.7 7.8 0s5.4 1.7 8.2 0 4.6-1.6 6 -.6" fill="none" stroke="#6cc6dc" stroke-width="2" stroke-linecap="round"/>
<path d="M4.5 23.5h11.2c1.6 0 2.6.5 3.5 1.6l2.6 3.2" fill="none" stroke="#e8eef5" stroke-opacity=".6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="10.4" cy="18" r="2" fill="#f6bb5c"/>
<path d="M14 15.6a4 4 0 0 1 0 4.8" fill="none" stroke="#f6bb5c" stroke-opacity=".75" stroke-width="1.6" stroke-linecap="round"/>
<path d="M16.7 14.1a7 7 0 0 1 0 7.8" fill="none" stroke="#f6bb5c" stroke-opacity=".4" stroke-width="1.6" stroke-linecap="round"/>
</svg>`;

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 1 });
async function png(size, pad = 0) {
  await page.setViewportSize({ width: size, height: size });
  await page.setContent(`<html><body style="margin:0;background:transparent">${markSvg(pad).replace("<svg ", `<svg width="${size}" height="${size}" `)}</body></html>`);
  return page.screenshot({ omitBackground: true, clip: { x: 0, y: 0, width: size, height: size } });
}
const out = (p, b) => (mkdirSync(dirname(join(root, p)), { recursive: true }), writeFileSync(join(root, p), b));

// favicon.ico: PNG-compressed entries (supported by every current browser)
const sizes = [16, 32, 48];
const pngs = [];
for (const s of sizes) pngs.push(await png(s));
const head = Buffer.alloc(6 + 16 * sizes.length);
head.writeUInt16LE(0, 0); head.writeUInt16LE(1, 2); head.writeUInt16LE(sizes.length, 4);
let offset = head.length;
sizes.forEach((s, i) => {
  const e = 6 + 16 * i;
  head.writeUInt8(s, e); head.writeUInt8(s, e + 1); head.writeUInt8(0, e + 2); head.writeUInt8(0, e + 3);
  head.writeUInt16LE(1, e + 4); head.writeUInt16LE(32, e + 6);
  head.writeUInt32LE(pngs[i].length, e + 8); head.writeUInt32LE(offset, e + 12);
  offset += pngs[i].length;
});
out("src/app/favicon.ico", Buffer.concat([head, ...pngs]));
out("src/app/icon.svg", markSvg());
out("src/app/apple-icon.png", await png(180, 3));
out("public/icons/icon-192.png", await png(192));
out("public/icons/icon-512.png", await png(512));
out("public/icons/maskable-512.png", await png(512, 6));
await browser.close();
console.log("brand assets written");
