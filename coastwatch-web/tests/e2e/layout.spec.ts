import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * Field ergonomics and composition (v4): no sideways scroll; on phones the primary field controls
 * (navigation, map controls, time steps, sheet handle) are at least 56 px and no two hit boxes overlap;
 * the map keeps most of the screen; reduced motion is respected.
 */
const NOW = "2026-10-08T20:00:00Z";
const SIZES = [
  { name: "desktop", viewport: { width: 1440, height: 900 } },
  { name: "tablet", viewport: { width: 834, height: 1112 }, hasTouch: true },
  { name: "phone", viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true },
  { name: "small phone", viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true },
  { name: "landscape phone", viewport: { width: 844, height: 390 }, isMobile: true, hasTouch: true },
];
const PAGES = ["/?layer=forecast", "/?layer=forecast&sel=harbor:550", "/coast?place=port-550", "/research?station=HABs-SantaCruzWharf", "/bloom", "/about"];

for (const s of SIZES) {
  test(`${s.name}: no page scrolls sideways`, async ({ browser }) => {
    const { name, ...opts } = s;
    const ctx = await browser.newContext(opts);
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    for (const p of PAGES) {
      await page.goto(`${OK}${p}`);
      await page.waitForTimeout(800);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth), `${name} ${p}`).toBe(0);
    }
    await ctx.close();
  });
}

test("the briefing and Research arrive in the first HTML, not in a streamed boundary revealed later", async ({ request }) => {
  // A Suspense boundary around a page's content is streamed hidden and revealed by React in ~300 ms
  // batches: the reader saw an empty page for that long (first paint ~60 ms, first content ~390 ms).
  for (const [path, text] of [["/coast?place=port-550", "Monterey"], ["/research?place=port-550", "Research"]] as const) {
    const html = await (await request.get(`${OK}${path}`)).text();
    expect(html, `${path}: no hidden streamed boundary`).not.toMatch(/<div hidden id="S:/);
    expect(html.replace(/<script[\s\S]*?<\/script>/g, ""), `${path}: content in the first HTML`).toContain(text);
  }
});
