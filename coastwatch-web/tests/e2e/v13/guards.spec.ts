import { expect, test, type Browser, type Page } from "@playwright/test";
import { OK } from "../ports";
import { MOSS, portRecord, RETURNING, seed } from "./helpers";

/**
 * Guards for the whole v13 shell (the independent review, audit F): things no single page's suite owns.
 *
 *  G1  A phone never throws its page away after hydration. v12's margin rendered a different element on a phone
 *      once the media query was known, so every route was unmounted and built again (effects twice, motion
 *      replayed, focus and typing lost); that is what failed research.spec.ts:154 on the slower Linux runner.
 *  G2  At 390 and 320 px every control of a page can be reached by scrolling alone: wholly between the top bar
 *      and the destinations bar, and not under anything. v12's My Coast kept "Save this trip" under the tab bar.
 *  G3  Hand-offs arrive: what one page promises in a link, the next page shows.
 *
 * A slowed CPU (4x) keeps the window between first paint and hydration wide enough to catch a remount.
 */
test.use({ storageState: { cookies: [], origins: [] } });

/**
 * Defects open at the review of f50375db3 (docs/coastwatch/v13/implementation/review.md). Each is expected to
 * FAIL until fixed: Playwright reports the test as passed while it fails and as failed once it passes, so the
 * entry must be deleted with the fix and the guard then holds the page to it.
 *  F-01  chart-mode pages on a phone: at the sheet's resting height its scroller ends below the destinations
 *        bar, so the last controls cannot be scrolled into view (Today's rows, Notices' links, Explore's layer
 *        switch and time rail). Needs "More" or a drag first.
 *  F-07  the trip workspace at 320 px: the last links sit under the fixed result bar at the end of the page.
 */
// F-01 is fixed in the shell (the sheet's scroller ends where the visible sheet ends at every detent): its five
// entries are gone, so the guard now holds those pages to it.
const KNOWN: Record<string, string> = {};

const PHONES = [
  { width: 390, height: 844 },
  { width: 320, height: 640 },
];

// route, and the element that is the page (present in the server's HTML)
const ROUTES: Array<[string, string, string]> = [
  ["Today", `/today?place=${MOSS}`, "nx-margin"],
  ["Explore", `/explore?place=${MOSS}`, "nx-explore"],
  ["My Coast", `/coast?place=${MOSS}`, "nx-coast"],
  ["My Coast, trip", `/coast?place=${MOSS}&view=trip`, "nx-coast"],
  ["Research", `/research?station=HABs-SantaCruzWharf&var=pDA`, "nx-research"],
  ["Alerts", `/alerts?place=${MOSS}`, "nx-alerts"],
  ["Your data", `/account`, "nx-account"],
  ["Notices", `/notices?place=${MOSS}`, "nx-margin"],
];

async function phone(browser: Browser, size: { width: number; height: number }, cpu = 1) {
  const ctx = await browser.newContext({ viewport: size, isMobile: true, hasTouch: true, storageState: { cookies: [], origins: [] } });
  await seed(ctx, { ...RETURNING, "nx-port-v1": portRecord({ home: MOSS }), "nx-coast-v1": JSON.stringify({ fishery: "dungeness_commercial" }) });
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  if (cpu > 1) {
    const cdp = await ctx.newCDPSession(page);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpu });
  }
  return { ctx, page, errors };
}

test.describe("G1: no remount after hydration on a phone", () => {
  for (const [name, path, root] of ROUTES) {
    test(`${name} at 390 px, CPU 4x: the page element the server sent is the one still on screen`, async ({ browser }) => {
      const { ctx, page, errors } = await phone(browser, PHONES[0], 4);
      // the first node with the page's test id, as parsed from the server's HTML, before any script of the app runs
      await page.addInitScript((id) => {
        const w = window as unknown as { __first?: Element | null; __replaced?: number };
        w.__replaced = 0;
        new MutationObserver(() => {
          const el = document.querySelector(`[data-testid="${id}"]`);
          if (!el) return;
          if (!w.__first) w.__first = el;
          else if (el !== w.__first) w.__replaced = (w.__replaced ?? 0) + 1;
        }).observe(document, { childList: true, subtree: true });
      }, root);
      await page.goto(`${OK}${path}`);
      await expect(page.getByTestId(root)).toBeVisible();
      // hydration and the first data are long done at 4x after this
      await page.waitForTimeout(6000);
      const r = await page.evaluate((id) => {
        const w = window as unknown as { __first?: Element | null; __replaced?: number };
        return { had: !!w.__first, connected: w.__first?.isConnected ?? false, same: w.__first === document.querySelector(`[data-testid="${id}"]`), replaced: w.__replaced };
      }, root);
      expect(r, `${name}: the server-rendered page element must survive hydration`).toEqual({ had: true, connected: true, same: true, replaced: 0 });
      expect(errors).toEqual([]);
      await ctx.close();
    });
  }
});

/**
 * Every control inside the page, scrolled to by the browser's own scrolling (no sheet drag): is all of it between
 * the bars, and is it the thing under its own centre? Returns the ones that are not, in words.
 */
async function unreachable(page: Page, root: string): Promise<string[]> {
  return page.evaluate(async (id) => {
    const top = document.querySelector('[data-testid="nx-bar"]')?.getBoundingClientRect().bottom ?? 0;
    const tabs = document.querySelector('[data-testid="nx-tabs"]');
    const tabsTop = tabs && getComputedStyle(tabs).display !== "none" && tabs.getBoundingClientRect().height > 0 ? tabs.getBoundingClientRect().top : innerHeight;
    const scope = document.querySelector(`[data-testid="${id}"]`);
    if (!scope) return [`no [data-testid=${id}]`];
    const bad: string[] = [];
    const els = [...scope.querySelectorAll<HTMLElement>("button, a[href], input, select, textarea, [role=radio], [role=slider]")].filter((el) => {
      if ((el as HTMLButtonElement).disabled || el.closest("[inert], [aria-hidden=true]")) return false;
      const cs = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      return cs.visibility !== "hidden" && cs.display !== "none" && r.width > 0 && r.height > 0 && !el.closest("details:not([open]) > :not(summary)");
    });
    for (const el of els) {
      const name = `${el.tagName.toLowerCase()} “${(el.getAttribute("aria-label") ?? el.innerText ?? (el as HTMLInputElement).name ?? "").replace(/\s+/g, " ").trim().slice(0, 48)}”${el.dataset.testid ? ` [${el.dataset.testid}]` : ""}`;
      // any of the browser's own alignments will do: a reader scrolls until the control is clear of the bars
      let why = "";
      for (const block of ["center", "start", "end"] as const) {
        el.scrollIntoView({ block, inline: "nearest", behavior: "instant" });
        await new Promise((r) => requestAnimationFrame(() => r(null)));
        const r = el.getBoundingClientRect();
        // a control taller than the room between the bars cannot be wholly inside it; its top must be
        const h = Math.min(r.height, tabsTop - top);
        const cx = r.left + r.width / 2;
        if (r.top < top - 0.5 || r.top + h > tabsTop + 0.5) why = `at y ${Math.round(r.top)}–${Math.round(r.bottom)}, outside ${Math.round(top)}–${Math.round(tabsTop)}`;
        // its centre on the screen (a drag handle at the end of a track may overhang the edge by half its width)
        else if (cx < 0 || cx > innerWidth) why = `off the side (x ${Math.round(r.left)}–${Math.round(r.right)} of ${innerWidth})`;
        else {
          const hit = document.elementFromPoint(cx, r.top + h / 2);
          // the control itself, something inside it, or its label
          const own = !hit || el === hit || el.contains(hit) || hit.contains(el) || hit.closest("label")?.contains(el) || (hit.closest("label") as HTMLLabelElement | null)?.control === el;
          why = own ? "" : `under ${hit!.tagName.toLowerCase()}${typeof hit!.className === "string" && hit!.className ? `.${hit!.className.split(" ")[0]}` : ""} “${(hit as HTMLElement).innerText?.replace(/\s+/g, " ").slice(0, 30) ?? ""}”`;
        }
        if (!why) break;
      }
      if (why) bad.push(`${name}: ${why}`);
    }
    return bad;
  }, root);
}

test.describe("G2: every control is reachable by scrolling alone", () => {
  for (const size of PHONES)
    for (const [name, path, root] of ROUTES) {
      test(`${name} at ${size.width} px`, async ({ browser }) => {
        const known = KNOWN[`${name} at ${size.width} px`];
        test.fail(!!known, `open defect ${known}`);
        const { ctx, page } = await phone(browser, size);
        await page.goto(`${OK}${path}`);
        await expect(page.getByTestId(root)).toBeVisible();
        await page.waitForTimeout(3500); // the port's records, the sea window and the chart have arrived
        expect(await unreachable(page, root), `${name} at ${size.width} px`).toEqual([]);
        // and the page itself never scrolls sideways
        expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
        await ctx.close();
      });
    }

  test("the owner's trip at 320 px: Save as new, Update and the saved list are all reached by scrolling, with the numbers filled in", async ({ browser }) => {
    const { ctx, page } = await phone(browser, PHONES[1]);
    await page.goto(`${OK}/coast?place=${MOSS}&view=trip`);
    await expect(page.getByTestId("trip-inputs")).toBeVisible();
    for (const [label, v] of [
      ["Catch", "400"],
      ["Price", "9"],
      ["Fuel", "30"],
      ["Fuel price", "8.24"],
      ["Bait and ice", "100"],
      ["Fees", "0"],
      ["Crew", "1"],
    ]) {
      const f = page.getByRole("textbox", { name: label, exact: true });
      await f.scrollIntoViewIfNeeded();
      await f.fill(v);
    }
    await page.getByRole("radio", { name: "After costs" }).click();
    await page.getByRole("textbox", { name: "Crew share", exact: true }).fill("20");
    await page.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }).click();
    await page.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Shared" }).click();
    await expect(page.getByTestId("trip-netbar-value")).toContainText("$2,602.24");
    // a real tap (Playwright refuses a covered or off-screen target), not a programmatic click
    await page.getByTestId("trip-save-new").tap();
    await expect(page.getByTestId("trip-said")).toContainText("Saved as");
    // open defect F-07: "Copy numbers-only link" ends under the fixed result bar; everything else must be clear
    expect((await unreachable(page, "nx-coast")).filter((x) => !/\[trip-copy-link\]/.test(x))).toEqual([]);
    await ctx.close();
  });
});

test.describe("G3: hand-offs arrive", () => {
  test("Explore → Research: the nearby record opens that station (within 25 km) on the quantity the layer shows", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
    await seed(ctx, { ...RETURNING, "nx-port-v1": portRecord({ home: MOSS }) });
    const page = await ctx.newPage();
    // C-HARM at a cell in the bay (the cell v13/explore.spec.ts reads): a pier that publishes pDA is offered
    await page.goto(`${OK}/explore?place=${MOSS}&layer=charm&at=36.85,-122.00`);
    const link = page.getByTestId("nx-record-link");
    await expect(link).toBeVisible({ timeout: 20_000 });
    const m = /^See the record nearby: (.+), ([\d.]+) km$/.exec((await link.innerText()).trim())!;
    expect(m, "the link names the station and its distance").not.toBeNull();
    expect(Number(m[2])).toBeLessThanOrEqual(25);
    const href = (await link.getAttribute("href"))!;
    const q = new URL(href, OK).searchParams;
    expect(q.get("var")).toBe("pDA");
    await link.click();
    await expect(page).toHaveURL(/\/research\?/);
    // Research shows that station and that quantity, not its default
    await expect(page.getByTestId("research-title")).toHaveText(m[1]);
    await expect(page.getByTestId("nx-research")).toHaveAttribute("data-station", q.get("station")!);
    await expect(page.getByTestId("rail-pDA")).toHaveAttribute("aria-checked", "true");
    // the port travels, and the dates the link carried are the window
    const after = new URL(page.url()).searchParams;
    expect(after.get("place")).toBe(MOSS);
    expect(after.get("from")).toBe(q.get("from"));
    expect(after.get("to")).toBe(q.get("to"));
    await ctx.close();
  });

  test("Research honours a quantity other than pDA when a link names it", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
    await seed(ctx, RETURNING);
    const page = await ctx.newPage();
    await page.goto(`${OK}/research?station=HABs-SantaCruzWharf&var=temp&place=${MOSS}`);
    await expect(page.getByTestId("rail-temp")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("rail-pDA")).toHaveAttribute("aria-checked", "false");
    await expect(page.getByTestId("research-chart")).toContainText("Water temperature");
    await ctx.close();
  });

  test("Explore → My Coast: the cell that was read and its date are shown on My Coast, and survive into the trip workspace", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
    await seed(ctx, { ...RETURNING, "nx-port-v1": portRecord({ home: MOSS }), "nx-coast-v1": JSON.stringify({ fishery: "dungeness_commercial" }) });
    const page = await ctx.newPage();
    await page.goto(`${OK}/explore?place=${MOSS}&layer=chl&at=36.87,-122.02`);
    const link = page.getByTestId("nx-coast-link");
    await expect(link).toBeVisible({ timeout: 20_000 });
    const q = new URL((await link.getAttribute("href"))!, OK).searchParams;
    const [lat, lon] = q.get("at")!.split(",").map(Number);
    await link.click();
    await expect(page.getByTestId("coast-title")).toHaveText("A trip from Moss Landing");
    // the point is said in degrees and minutes, with the date it was read for
    const header = page.locator("[data-nx-desk-header]");
    const deg = (v: number) => `${Math.floor(Math.abs(v))}°`;
    await expect(header).toContainText(deg(lat));
    await expect(header).toContainText(deg(lon));
    await expect(header).toContainText(/Oct \d+/);
    await page.getByTestId("coast-to-trip").click();
    await expect(page.getByTestId("trip-inputs")).toBeVisible();
    expect(new URL(page.url()).searchParams.get("at")).toBe(q.get("at"));
    await expect(page.locator("[data-nx-desk-header]")).toContainText(deg(lat));
    await ctx.close();
  });

  test("Today's gate links land on pages that show the port they were opened from", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
    // home is Bodega Bay; Today is opened at Moss Landing by a link: every onward page must stay at Moss Landing
    await seed(ctx, { ...RETURNING, "nx-port-v1": portRecord({ home: "port-473", last: "port-473" }), "nx-coast-v1": JSON.stringify({ fishery: "dungeness_commercial" }) });
    const page = await ctx.newPage();
    for (const [gate, link, where, says] of [
      ["gate-domoic", /Research/, /\/research\?place=port-592&var=pDA/, ["research-status", /./]],
      ["gate-sea", /Explore/, /\/explore\?place=port-592/, ["nx-port-control", /Moss Landing/]],
      ["gate-sea", /trip|My Coast/i, /\/coast\?place=port-592&view=trip/, ["coast-title", /Moss Landing/]],
      ["gate-whale", /alert/i, /\/alerts\?place=port-592/, ["nx-alerts", /Moss Landing/]],
    ] as const) {
      await page.goto(`${OK}/today?place=${MOSS}`);
      await expect(page.getByTestId("gate-sea")).toContainText("Waves", { timeout: 15_000 });
      const a = page.getByTestId("gate-list").locator(`a[href*="place=${MOSS}"]`).filter({ hasText: link }).first();
      await expect(a, `${gate}: a link matching ${link}`).toBeVisible();
      await a.click();
      await expect(page).toHaveURL(where);
      await expect(page.getByTestId(says[0])).toContainText(says[1]);
      await expect(page.getByTestId("nx-port-control")).toContainText("Moss Landing");
    }
    await ctx.close();
  });
});
