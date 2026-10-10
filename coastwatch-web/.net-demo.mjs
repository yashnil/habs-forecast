import { chromium } from "@playwright/test";
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
const bad = [];
p.on("response", (r) => r.status() >= 400 && bad.push(r.status() + " " + r.url()));
await p.goto(process.argv[2], { waitUntil: "networkidle" }); await p.waitForTimeout(2000);
console.log(bad.length, "failed"); console.log(bad.slice(0, 6).join("\n"));
await b.close();
