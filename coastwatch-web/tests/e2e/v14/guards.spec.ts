import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";
import { FRAMES, MOSS, PHONES, SIZES, context, fillRegressionTrip, isPhone, sideways } from "./helpers";

/**
 * Guards for the v14 shell (the synthesis restyle): what no single page's suite owns. The v13 guards (G1–G3,
 * tests/e2e/v13/guards.spec.ts) carried to the new layout, and four more (G4–G7).
 *
 *  G1  A phone never throws its page away after hydration (the server's page element is the one on screen).
 *  G2  At 390 and 320 px every control of a page can be reached by scrolling alone: wholly between the top bar and
 *      the destinations bar, and the thing under its own centre.
 *  G3  Hand-offs arrive: Explore → Research and → My Coast, and Today's links, keep the port they were opened from.
 *  G4  Nothing runs off the sides of the screen at the five review sizes, on the five frames' pages.
 *  G5  With reduced motion, a settled page runs no animation and asks for no animation frames.
 *  G6  Each map layer chip turns its layer on and off on the chart itself.
 *  G7  The app at the five frames has no page errors (the same states scripts/v14-app-shots.mjs photographs).
 *
 * Hooks these rely on (keep them through the restyle; the new ones are marked NEW):
 *   page roots   nx-margin (Today/Notices, in the server HTML), nx-today, nx-explore, nx-coast, nx-research, nx-alerts, nx-account
 *   shell        nx-bar (the top bar or floating brand/nav bar: its bottom is the top of the room a page scrolls in),
 *                nx-tabs (the phone destinations bar), nx-port-control (names the current port)
 *   hand-offs    nx-record-link, nx-coast-link (Explore), coast-title, coast-to-trip, research-title, rail-<var>,
 *                [data-nx-desk-header] (My Coast), the Today links themselves (any <a> to /research, /coast?…view=trip,
 *                /alerts carrying place=)
 *   NEW (D)      the layer chips: a group named "Map layers" (role=group, aria-label) of buttons with aria-pressed and
 *                data-layer="<id>"; on a phone a button named "Layers" opens it. The chart host carries
 *                data-layers="<ids drawn, space-separated>", set from what the map actually draws (the style's layer
 *                visibility after the change), never copied from the chip's own state.
 *   Today        the gate detail links sit in the ticket's Record tab (role=tab, name "Record"), mounted when chosen.
 *   G6 note      the CDPH test reading draws only with the Toxin tests chip on (off by default on Today): any test of
 *                that reading must turn the chip on first. G6 itself toggles every chip both ways from its default.
 */
test.use({ storageState: { cookies: [], origins: [] } });

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

test.describe("G1: no remount after hydration on a phone", () => {
  for (const [name, path, root] of ROUTES) {
    test(`${name} at 390 px, CPU 4x: the page element the server sent is the one still on screen`, async ({ browser }) => {
      test.setTimeout(90_000); // a 4x-slowed CPU on a busy machine
      const { ctx, page, errors } = await context(browser, PHONES[0]);
      const cdp = await ctx.newCDPSession(page);
      await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
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

/** Every control inside the page, scrolled to by the browser's own scrolling (no sheet drag), clear of the bars and not covered. */
async function unreachable(page: Page, root: string): Promise<string[]> {
  return page.evaluate(async (id) => {
    const top = document.querySelector('[data-testid="nx-bar"]')?.getBoundingClientRect().bottom ?? 0;
    const tabs = document.querySelector('[data-testid="nx-tabs"]');
    const tabsTop = tabs && getComputedStyle(tabs).display !== "none" && tabs.getBoundingClientRect().height > 0 ? tabs.getBoundingClientRect().top : innerHeight;
    const scope = document.querySelector(`[data-testid="${id}"]`);
    if (!scope) return [`no [data-testid=${id}]`];
    const bad: string[] = [];
    const els = [...scope.querySelectorAll<HTMLElement>("button, a[href], input, select, textarea, [role=radio], [role=slider], [role=switch], [role=tab]")].filter((el) => {
      if ((el as HTMLButtonElement).disabled || el.closest("[inert], [aria-hidden=true]")) return false;
      const cs = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      return cs.visibility !== "hidden" && cs.display !== "none" && r.width > 0 && r.height > 0 && !el.closest("details:not([open]) > :not(summary)");
    });
    for (const el of els) {
      const name = `${el.tagName.toLowerCase()} “${(el.getAttribute("aria-label") ?? el.innerText ?? (el as HTMLInputElement).name ?? "").replace(/\s+/g, " ").trim().slice(0, 48)}”${el.dataset.testid ? ` [${el.dataset.testid}]` : ""}`;
      let why = "";
      for (const block of ["center", "start", "end"] as const) {
        el.scrollIntoView({ block, inline: "nearest", behavior: "instant" });
        await new Promise((r) => requestAnimationFrame(() => r(null)));
        const r = el.getBoundingClientRect();
        const h = Math.min(r.height, tabsTop - top);
        const cx = r.left + r.width / 2;
        if (r.top < top - 0.5 || r.top + h > tabsTop + 0.5) why = `at y ${Math.round(r.top)}–${Math.round(r.bottom)}, outside ${Math.round(top)}–${Math.round(tabsTop)}`;
        else if (cx < 0 || cx > innerWidth) why = `off the side (x ${Math.round(r.left)}–${Math.round(r.right)} of ${innerWidth})`;
        else {
          const hit = document.elementFromPoint(cx, r.top + h / 2);
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
        const { ctx, page } = await context(browser, size);
        await page.goto(`${OK}${path}`);
        await expect(page.getByTestId(root)).toBeVisible();
        await page.waitForTimeout(3500);
        expect(await unreachable(page, root), `${name} at ${size.width} px`).toEqual([]);
        await ctx.close();
      });
    }

  test("the regression trip at 320 px: Save as new, Update and the saved list are all reached by scrolling, with the numbers filled in", async ({ browser }) => {
    const { ctx, page } = await context(browser, PHONES[1]);
    await page.goto(`${OK}/coast?place=${MOSS}&view=trip`);
    await expect(page.getByTestId("trip-inputs")).toBeVisible();
    await fillRegressionTrip(page);
    await expect(page.locator('[data-testid="trip-netbar-value"], [data-testid="trip-net"]').first()).toContainText("$2,602.24");
    await page.getByTestId("trip-save-new").tap();
    await expect(page.getByTestId("trip-said")).toContainText("Saved as");
    expect(await unreachable(page, "nx-coast")).toEqual([]);
    await ctx.close();
  });
});

test.describe("G3: hand-offs arrive", () => {
  const desk = SIZES[0];

  test("Explore → Research: the nearby record opens that station (within 25 km) on the quantity the layer shows", async ({ browser }) => {
    const { ctx, page } = await context(browser, desk);
    await page.goto(`${OK}/explore?place=${MOSS}&layer=charm&at=36.85,-122.00`);
    const link = page.getByTestId("nx-record-link");
    await expect(link).toBeVisible({ timeout: 20_000 });
    const m = /^See the record nearby: (.+), ([\d.]+) km$/.exec((await link.innerText()).replace(/\s+/g, " ").trim());
    expect(m, "the link names the station and its distance").not.toBeNull();
    expect(Number(m![2])).toBeLessThanOrEqual(25);
    const q = new URL((await link.getAttribute("href"))!, OK).searchParams;
    expect(q.get("var")).toBe("pDA");
    await link.click();
    await expect(page).toHaveURL(/\/research\?/);
    await expect(page.getByTestId("research-title")).toHaveText(m![1]);
    await expect(page.getByTestId("nx-research")).toHaveAttribute("data-station", q.get("station")!);
    await expect(page.getByTestId("rail-pDA")).toHaveAttribute("aria-checked", "true");
    const after = new URL(page.url()).searchParams;
    expect(after.get("place")).toBe(MOSS);
    expect(after.get("from")).toBe(q.get("from"));
    expect(after.get("to")).toBe(q.get("to"));
    await expect(page.getByTestId("nx-port-control")).toContainText("Moss Landing");
    await ctx.close();
  });

  test("Explore → My Coast: the cell that was read and its date reach My Coast and survive into the trip workspace", async ({ browser }) => {
    const { ctx, page } = await context(browser, desk);
    await page.goto(`${OK}/explore?place=${MOSS}&layer=chl&at=36.87,-122.02`);
    const link = page.getByTestId("nx-coast-link");
    await expect(link).toBeVisible({ timeout: 20_000 });
    const q = new URL((await link.getAttribute("href"))!, OK).searchParams;
    const [lat, lon] = q.get("at")!.split(",").map(Number);
    await link.click();
    await expect(page.getByTestId("coast-title")).toContainText("Moss Landing");
    const header = page.locator("[data-nx-desk-header]");
    const deg = (v: number) => `${Math.floor(Math.abs(v))}°`;
    await expect(header).toContainText(deg(lat));
    await expect(header).toContainText(deg(lon));
    await expect(header).toContainText(/Oct \d+/);
    await page.getByTestId("coast-to-trip").click();
    await expect(page.getByTestId("trip-inputs")).toBeVisible();
    expect(new URL(page.url()).searchParams.get("at")).toBe(q.get("at"));
    await ctx.close();
  });

  test("Today's links land on pages that show the port they were opened from (not the home port)", async ({ browser }) => {
    // home is Bodega Bay; Today is opened at Moss Landing by a link: every onward page must stay at Moss Landing
    const { ctx, page } = await context(browser, desk, { home: "port-473" });
    await page.goto(`${OK}/today?place=${MOSS}`);
    await expect(page.getByTestId("nx-today")).toBeVisible();
    await page.waitForTimeout(2500);
    const onward = () =>
      page
        .getByTestId("nx-today")
        .locator("a[href]")
        .evaluateAll((as) => [...new Set(as.map((a) => a.getAttribute("href")!).filter((h) => /^\/(research|explore|coast|alerts|notices)\b/.test(h)))]);
    // at load: the ticket's own links (samples nearby, plan a trip) already keep the port
    const atLoad = await onward();
    // the gate detail links (alerts for this port, Explore, notices) live in the ticket's Record tab, mounted when chosen
    const record = () => page.getByTestId("nx-today").getByRole("tab", { name: "Record" });
    await record().click();
    await expect(record()).toHaveAttribute("aria-selected", "true");
    const hrefs = [...new Set([...atLoad, ...(await onward())])];
    // every onward link names the port it came from
    for (const h of hrefs) expect(new URL(h, OK).searchParams.get("place"), `${h} keeps the port`).toBe(MOSS);
    // and the three hand-offs the page promises are there
    for (const want of [/^\/research\?/, /^\/coast\?.*view=trip/, /^\/alerts\?/]) expect(hrefs.some((h) => want.test(h)), `a Today link matching ${want}: ${hrefs.join(" ")}`).toBe(true);
    for (const h of hrefs) {
      await page.goto(`${OK}/today?place=${MOSS}`);
      await expect(page.getByTestId("nx-today")).toBeVisible();
      if (!atLoad.includes(h)) await record().click();
      const a = page.getByTestId("nx-today").locator(`a[href="${h}"]`).first();
      await a.scrollIntoViewIfNeeded();
      await a.click();
      await expect(page).toHaveURL(new RegExp(`place=${MOSS}`));
      await expect(page.getByTestId("nx-port-control"), `${h} shows Moss Landing`).toContainText("Moss Landing");
    }
    await ctx.close();
  });
});

test.describe("G4: nothing runs off the sides of the screen", () => {
  for (const size of SIZES)
    for (const f of FRAMES) {
      test(`${f.name} at ${size.width}×${size.height}`, async ({ browser }) => {
        const { ctx, page, errors } = await context(browser, size, { reducedMotion: "reduce" });
        await f.open(page);
        await page.waitForTimeout(2000);
        expect(await sideways(page), `${f.name} at ${size.width}×${size.height}`).toEqual([]);
        // G7 rides along: the frame's state is reached without a page error
        expect(errors).toEqual([]);
        await ctx.close();
      });
    }
});

test.describe("G5: reduced motion leaves nothing running", () => {
  for (const size of [SIZES[0], PHONES[0]])
    for (const f of FRAMES) {
      test(`${f.name} at ${size.width} px`, async ({ browser }) => {
        const { ctx, page } = await context(browser, size, { reducedMotion: "reduce" });
        await page.addInitScript(() => {
          const w = window as unknown as { __raf: number };
          w.__raf = 0;
          const raf = window.requestAnimationFrame.bind(window);
          window.requestAnimationFrame = (cb) => {
            w.__raf++;
            return raf(cb);
          };
        });
        await f.open(page);
        // data and tiles may still be arriving: wait for a quiet 1.5 s window (a loop that never stops never gets one)
        let last = { raf: -1, running: [] as string[] };
        const deadline = Date.now() + 20_000;
        while (Date.now() < deadline) {
          const before = await page.evaluate(() => (window as unknown as { __raf: number }).__raf);
          await page.waitForTimeout(1500);
          last = await page.evaluate((b) => {
            const running = document
              .getAnimations()
              .filter((a) => a.playState === "running")
              .map((a) => {
                const t = (a.effect as KeyframeEffect | null)?.target as Element | null;
                const name = (a as CSSAnimation).animationName ?? (a as CSSTransition).transitionProperty ?? a.id ?? "animation";
                return `${name} on ${t ? `${t.tagName.toLowerCase()}${t.id ? `#${t.id}` : ""}${typeof t.className === "string" && t.className ? `.${t.className.split(" ")[0]}` : ""}` : "?"}`;
              });
            return { raf: (window as unknown as { __raf: number }).__raf - b, running };
          }, before);
          if (last.raf === 0 && last.running.length === 0) break;
        }
        expect(last.running, "no CSS or Web Animations still running").toEqual([]);
        expect(last.raf, "no animation frames requested by a settled page").toBe(0);
        await ctx.close();
      });
    }
});

test.describe("G6: the map layer chips toggle their layers on the chart", () => {
  const pages: Array<[string, string]> = [
    ["Today", `/today?place=${MOSS}`],
    ["My Coast, trip", `/coast?place=${MOSS}&view=trip`],
    ["Research", `/research?station=HABs-SantaCruzWharf&var=pDA`],
  ];
  for (const size of [SIZES[0], PHONES[0]])
    for (const [name, path] of pages) {
      test(`${name} at ${size.width} px`, async ({ browser }) => {
        const { ctx, page } = await context(browser, size);
        await page.goto(`${OK}${path}`);
        const host = page.locator("[data-layers]").first();
        await expect(host, "the chart host says which layers it draws (data-layers)").toBeAttached({ timeout: 20_000 });
        const group = page.getByRole("group", { name: "Map layers" });
        if (isPhone(size) && !(await group.isVisible().catch(() => false))) await page.getByRole("button", { name: "Layers", exact: true }).click();
        await expect(group).toBeVisible();
        const chips = group.locator("button[data-layer][aria-pressed]");
        const n = await chips.count();
        expect(n, "at least the four layers of the synthesis").toBeGreaterThanOrEqual(4);
        const drawn = async () => ((await host.getAttribute("data-layers")) ?? "").split(/\s+/).filter(Boolean);
        for (let i = 0; i < n; i++) {
          const chip = chips.nth(i);
          const id = (await chip.getAttribute("data-layer"))!;
          const was = (await chip.getAttribute("aria-pressed")) === "true";
          expect(await drawn(), `${id}: the chip says ${was ? "on" : "off"}`).toEqual(was ? expect.arrayContaining([id]) : expect.not.arrayContaining([id]));
          await chip.click();
          await expect(chip).toHaveAttribute("aria-pressed", String(!was));
          await expect.poll(drawn, { message: `${id} ${was ? "leaves" : "joins"} the chart` }).toEqual(was ? expect.not.arrayContaining([id]) : expect.arrayContaining([id]));
          await chip.click();
          await expect(chip).toHaveAttribute("aria-pressed", String(was));
          await expect.poll(drawn, { message: `${id} is restored` }).toEqual(was ? expect.arrayContaining([id]) : expect.not.arrayContaining([id]));
        }
        await ctx.close();
      });
    }
});
