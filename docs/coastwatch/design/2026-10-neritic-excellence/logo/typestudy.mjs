import { chromium } from "../../../../../coastwatch-web/node_modules/@playwright/test/index.mjs";
const F = [["Hanken Grotesk",500],["Schibsted Grotesk",500],["Familjen Grotesk",500],["Onest",500],["Figtree",500],["Manrope",500],["Sora",400],["Bricolage Grotesque",500],["Newsreader",500],["Fraunces",500],["Literata",500],["Spectral",500],["Source Serif 4",600],["Instrument Serif",400],["IBM Plex Sans",500],["Libre Franklin",500]];
const fam = F.map(([f,w])=>`family=${f.replace(/ /g,"+")}:wght@${w}`).join("&");
let html = `<link href="https://fonts.googleapis.com/css2?${fam}&display=block" rel="stylesheet"><body style="margin:0;padding:24px;background:#fff;display:grid;grid-template-columns:repeat(4,1fr);gap:28px 20px">`;
for (const [f,w] of F) html += `<div><div style="font:${w} 54px '${f}';color:#0d1b2a;letter-spacing:-0.01em">neritic</div><div style="font:${w} 40px '${f}';color:#0d1b2a">Neritic</div><div style="font:12px monospace;color:#888">${f} ${w}</div></div>`;
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1300, height: 760 } });
await p.setContent(html); await p.waitForTimeout(2500); await p.screenshot({ path: "typestudy.png", fullPage: true }); await b.close();
