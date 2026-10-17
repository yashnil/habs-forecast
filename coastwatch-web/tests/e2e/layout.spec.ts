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

for (const w of [390, 320]) {
  test(`${w} px phone: field controls are 56 px and no hit boxes overlap`, async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: w, height: w === 320 ? 640 : 844 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/?layer=forecast`);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded(): boolean } }).__cwMap?.loaded());
    const r = await page.evaluate(() => {
      const box = (el: Element) => el.getBoundingClientRect();
      const field = [
        ...document.querySelectorAll('[data-testid="nav-dock"] a'),
        ...document.querySelectorAll('[data-testid="map-controls"] button'),
        ...document.querySelectorAll('[data-testid="timebar"] button'),
        ...document.querySelectorAll('[data-testid="sheet-handle"]'),
      ].filter((e) => box(e).width > 0);
      const small = field.filter((e) => box(e).height < 55.5 || box(e).width < 43.5).map((e) => `${e.getAttribute("aria-label") ?? (e as HTMLElement).innerText} ${Math.round(box(e).width)}x${Math.round(box(e).height)}`);
      const all = [...document.querySelectorAll("button, a[href], select, input")].filter((e) => {
        const b = box(e);
        const st = getComputedStyle(e);
        return b.width > 0 && b.height > 0 && st.visibility !== "hidden" && !e.closest("[inert],[aria-hidden=true]") && b.bottom > 0 && b.top < innerHeight && b.right > 0 && b.left < innerWidth;
      });
      const overlaps: string[] = [];
      for (let i = 0; i < all.length; i++)
        for (let j = i + 1; j < all.length; j++) {
          const a = all[i], b = all[j];
          if (a.contains(b) || b.contains(a)) continue;
          const A = box(a), B = box(b);
          const ox = Math.min(A.right, B.right) - Math.max(A.left, B.left);
          const oy = Math.min(A.bottom, B.bottom) - Math.max(A.top, B.top);
          if (ox > 1 && oy > 1) {
            // only a real overlap when one is drawn over the other at the shared point
            const x = Math.max(A.left, B.left) + ox / 2, y = Math.max(A.top, B.top) + oy / 2;
            const top = document.elementFromPoint(x, y);
            if (top && (a.contains(top) || b.contains(top)) && !(a.contains(top) && b.contains(top))) {
              const under = a.contains(top) ? b : a;
              const hit = document.elementsFromPoint(x, y).includes(under);
              if (hit) overlaps.push(`${(a as HTMLElement).innerText || a.getAttribute("aria-label")} / ${(b as HTMLElement).innerText || b.getAttribute("aria-label")}`);
            }
          }
        }
      return { small, overlaps };
    });
    expect(r.small, "field controls under 56 px").toEqual([]);
    expect(r.overlaps, "overlapping hit boxes").toEqual([]);
    await ctx.close();
  });
}

test("phone: the map keeps most of the screen with the sheet at its peek", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  await expect(page.getByTestId("mobile-sheet")).toHaveAttribute("data-snap", "peek");
  const sheet = (await page.getByTestId("mobile-sheet").boundingBox())!;
  const visibleSheet = 844 - sheet.y;
  expect(visibleSheet / 844).toBeLessThan(0.45);
  await ctx.close();
});

test("reduced motion: nothing animates on load or on selecting", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
  await expect(page.getByTestId("inspect-panel")).toBeVisible();
  const running = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === "running" && Number(a.effect?.getTiming().duration) > 10).length);
  expect(running).toBe(0);
  await ctx.close();
});

test("the briefing and Research arrive in the first HTML, not in a streamed boundary revealed later", async ({ request }) => {
  // A Suspense boundary around a page's content is streamed hidden and revealed by React in ~300 ms
  // batches: the reader saw an empty page for that long (first paint ~60 ms, first content ~390 ms).
  for (const [path, text] of [["/coast?place=port-550", "Monterey"], ["/research?place=port-550", "Research"]] as const) {
    const html = await (await request.get(`${OK}${path}`)).text();
    expect(html, `${path}: no hidden streamed boundary`).not.toMatch(/<div hidden id="S:/);
    expect(html.replace(/<script[\s\S]*?<\/script>/g, ""), `${path}: content in the first HTML`).toContain(text);
  }
});
