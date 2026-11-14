import { expect, test, type Browser, type Locator, type Page } from "@playwright/test";
import { OK } from "../ports";
import { MOSS, newContext, returning, seed, seedTrip } from "../v15/helpers";

/**
 * v15.1 free panels (DIRECTION §4, src/nx/ds/panels): on a wide screen the ticket, the docks and the pass are moved by
 * their grips independently (the other stays put), resized, collapsed and restored, reset together, kept on screen when
 * the screen shrinks and remembered on reload. On a phone nothing floats: the sheet reads in order, nothing overlaps.
 */
test.use({ storageState: { cookies: [], origins: [] } });

type Box = { x: number; y: number; width: number; height: number };

/** A context at Moss Landing with no fixed device clock (the fixture server pins the date): pages may be reloaded. */
async function coastContext(browser: Browser, size = { width: 1440, height: 900 }) {
  const ctx = await browser.newContext({ viewport: size, ...(size.width <= 640 ? { isMobile: true, hasTouch: true } : {}), storageState: { cookies: [], origins: [] } });
  await seed(ctx, returning);
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  return { ctx, page, errors };
}

/** The panel's box once it has arrived and stopped (its arrival springs and the camera's glide have settled). */
async function settled(l: Locator): Promise<Box> {
  await expect(l).toBeVisible({ timeout: 20_000 });
  await expect.poll(() => l.evaluate((n) => getComputedStyle(n).opacity), { timeout: 15_000 }).toBe("1");
  let last = "";
  for (let i = 0; i < 40; i++) {
    const b = await l.boundingBox();
    const k = JSON.stringify(b && [Math.round(b.x), Math.round(b.y), Math.round(b.width), Math.round(b.height)]);
    if (k === last) return b!;
    last = k;
    await l.page().waitForTimeout(250);
  }
  return (await l.boundingBox())!;
}

async function drag(page: Page, handle: Locator, dx: number, dy: number, during?: () => Promise<void>) {
  const b = (await handle.boundingBox())!;
  const x = b.x + b.width / 2, y = b.y + b.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let i = 1; i <= 10; i++) await page.mouse.move(x + (dx * i) / 10, y + (dy * i) / 10);
  await during?.();
  await page.mouse.up();
}

const near = (a: Box, b: Box, tol = 3) => {
  for (const k of ["x", "y", "width", "height"] as const) expect(Math.abs(a[k] - b[k]), `${k}: ${a[k]} vs ${b[k]}`).toBeLessThanOrEqual(tol);
};
const overlap = (a: Box, b: Box) => a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;

test("My Coast: the sea window and the ticket move, resize, collapse and reset independently, and are remembered", async ({ browser }) => {
  test.setTimeout(120_000);
  const { ctx, page, errors } = await coastContext(browser);
  await page.goto(`${OK}/coast?place=${MOSS}`);
  const ticket = page.locator('[data-panel="ticket"]');
  const sea = page.locator('[data-panel="sea-window"]');
  const t0 = await settled(ticket);
  const s0 = await settled(sea);
  expect(overlap(t0, s0), "the default layout keeps them apart").toBe(false);
  await expect(page.getByTestId("panel-reset")).toHaveCount(0);

  // the sea window moves; the ticket stays put (guides show while it nears the screen's edge; nothing snaps)
  await drag(page, sea.locator("[data-panel-grip]"), -s0.x + 26, -90, async () => {
    await expect(page.getByTestId("panel-guides")).toBeVisible();
  });
  const s1 = await settled(sea);
  expect(Math.round(s1.y - s0.y)).toBe(-90);
  expect(Math.round(s1.x)).toBe(26);
  near(await settled(ticket), t0);
  await expect(page.getByTestId("panel-guides")).toHaveCount(0);

  // the ticket moves; the sea window stays put
  await drag(page, ticket.locator("[data-panel-grip]"), -160, 24);
  const t1 = await settled(ticket);
  expect(Math.round(t1.x - t0.x)).toBe(-160);
  expect(Math.round(t1.y - t0.y)).toBe(24);
  near(await settled(sea), s1);

  // arrow keys move a focused grip
  await ticket.locator("[data-panel-grip]").focus();
  for (let i = 0; i < 3; i++) await page.keyboard.press("ArrowLeft");
  const t2 = await settled(ticket);
  expect(Math.round(t1.x - t2.x)).toBe(24);
  await expect(ticket.locator("[data-panel-grip]")).toHaveAccessibleName(/^Move /);

  // resize the sea window from its corner: wider and shorter, within limits
  await drag(page, sea.locator("[data-panel-resize]"), 60, -40);
  const s2 = await settled(sea);
  expect(Math.round(s2.width - s1.width)).toBe(60);
  expect(Math.round(s2.height - s1.height)).toBe(-40);
  near(await settled(ticket), t2);

  // collapse to the header and restore
  await sea.getByTestId("panel-collapse").click();
  await expect(sea).toHaveAttribute("data-panel-collapsed", "");
  const sc = await settled(sea);
  expect(sc.height).toBeLessThan(60);
  await expect(sea.getByText("Sea window", { exact: true }).first()).toBeVisible();
  await expect(sea.getByTestId("panel-collapse")).toHaveAttribute("aria-expanded", "false");
  await sea.getByTestId("panel-collapse").click();
  near(await settled(sea), s2);

  // remembered on reload
  await page.reload();
  near(await settled(ticket), t2);
  near(await settled(sea), s2);
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("nx-panels-v1:coast") ?? "{}").v)).toBe(1);

  // reset: both back where the page puts them, the layout forgotten
  await page.getByTestId("panel-reset").first().click();
  near(await settled(ticket), t0, 4);
  near(await settled(sea), s0, 4);
  await expect(page.getByTestId("panel-reset")).toHaveCount(0);
  expect(await page.evaluate(() => localStorage.getItem("nx-panels-v1:coast"))).toBeNull();
  expect(errors).toEqual([]);
  await ctx.close();
});

test("panels stay on screen when the screen shrinks from 1440 to 1100", async ({ browser }) => {
  const { ctx, page } = await coastContext(browser);
  await page.goto(`${OK}/coast?place=${MOSS}`);
  const ticket = page.locator('[data-panel="ticket"]');
  const sea = page.locator('[data-panel="sea-window"]');
  const t0 = await settled(ticket);
  await settled(sea);
  // the ticket to the far right, the sea window to the bottom right corner
  await drag(page, ticket.locator("[data-panel-grip]"), 1440 - (t0.x + t0.width), 0);
  await drag(page, sea.locator("[data-panel-grip]"), 600, 0);
  await page.setViewportSize({ width: 1100, height: 760 });
  for (const p of [ticket, sea]) {
    const b = await settled(p);
    expect(b.x).toBeGreaterThanOrEqual(0);
    expect(b.y).toBeGreaterThanOrEqual(0);
    expect(b.x + b.width).toBeLessThanOrEqual(1100);
    expect(b.y + b.height).toBeLessThanOrEqual(760);
    // its controls are in reach
    await expect(p.locator("[data-panel-grip]")).toBeInViewport();
    await expect(p.getByTestId("panel-collapse")).toBeInViewport();
  }
  await ctx.close();
});

test("the trip workspace: the departure dock rests beside the ticket and each moves alone", async ({ browser }) => {
  const seeded = await seedTrip({ id: "panels-1" });
  const { ctx, page, errors } = await newContext(browser, { width: 1440, height: 900 }, seeded.storage);
  await page.goto(`${OK}/coast/trips?id=panels-1`);
  const ticket = page.locator('[data-panel="ticket"]');
  const dock = page.locator('[data-panel="departure-windows"]');
  await expect(page.getByTestId("trip-workspace")).toBeVisible({ timeout: 20_000 });
  const t0 = await settled(ticket);
  const d0 = await settled(dock);
  expect(d0.x + d0.width, "the resting dock stops short of the resting ticket").toBeLessThanOrEqual(t0.x - 8);
  await drag(page, dock.locator("[data-panel-grip]"), 40, -120);
  const d1 = await settled(dock);
  expect(Math.round(d1.y - d0.y)).toBe(-120);
  near(await settled(ticket), t0);
  await drag(page, ticket.locator("[data-panel-grip]"), -60, 30);
  near(await settled(dock), d1);
  expect(errors).toEqual([]);
  await ctx.close();
});

test("Today: the sea dock and the ticket move independently", async ({ browser }) => {
  const { ctx, page } = await coastContext(browser, { width: 1280, height: 800 });
  await page.goto(`${OK}/today?place=${MOSS}`);
  const ticket = page.locator('[data-panel="ticket"]');
  const sea = page.locator('[data-panel="sea-next-four-days"]');
  const t0 = await settled(ticket);
  const s0 = await settled(sea);
  expect(overlap(t0, s0)).toBe(false);
  await drag(page, ticket.locator("[data-panel-grip]"), -300, 0);
  near(await settled(sea), s0);
  await ctx.close();
});

test("My Coast trip view: the full-width pass is not a free panel, and no Reset layout appears", async ({ browser }) => {
  const { ctx, page } = await coastContext(browser, { width: 1100, height: 760 });
  // an entry left by an earlier build for the pass is dropped, so Menu offers no reset either
  await page.goto(`${OK}/about`);
  await page.evaluate(() => localStorage.setItem("nx-panels-v1:coast", JSON.stringify({ v: 1, panels: { pass: { p: { ax: "l", dx: 0, ay: "t", dy: 0 } } } })));
  await page.goto(`${OK}/coast?place=${MOSS}&view=trip`);
  await expect(page.getByTestId("trip-pass")).toBeVisible({ timeout: 20_000 });
  await expect(page.locator("[data-panel]")).toHaveCount(0);
  await expect(page.locator("[data-panel-grip]")).toHaveCount(0);
  await expect(page.getByTestId("panel-reset")).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => localStorage.getItem("nx-panels-v1:coast"))).toBeNull();
  await ctx.close();
});

/** True when the point at the centre of `l` hits `l` itself (nothing floats over it). */
async function reachable(l: Locator) {
  const b = (await l.boundingBox())!;
  return l.evaluate((n, p) => {
    const hit = document.elementFromPoint(p.x, p.y);
    return !!hit && (hit === n || n.contains(hit));
  }, { x: b.x + b.width / 2, y: b.y + b.height / 2 });
}

const CHROME = ".n12-bar > *, .nx-layers, .n12-controls";
async function chromeBoxes(page: Page): Promise<Box[]> {
  return page.evaluate((sel) => [...document.querySelectorAll(sel)].map((e) => e.getBoundingClientRect()).filter((r) => r.width && r.height).map((r) => ({ x: r.left, y: r.top, width: r.width, height: r.height })), CHROME);
}

test("Today: a ticket dragged into the top-left corner stops clear of the bar and chips; grip, collapse and reset stay in reach", async ({ browser }) => {
  const { ctx, page, errors } = await coastContext(browser);
  await page.goto(`${OK}/today?place=${MOSS}`);
  const ticket = page.locator('[data-panel="ticket"]');
  await settled(ticket);
  await drag(page, ticket.locator("[data-panel-grip]"), -2000, -2000);
  const t = await settled(ticket);
  for (const c of await chromeBoxes(page)) expect(overlap(t, c), `ticket clear of ${JSON.stringify(c)}`).toBe(false);
  expect(t.y + t.height).toBeLessThanOrEqual(900);
  expect(await reachable(ticket.locator("[data-panel-grip]"))).toBe(true);
  expect(await reachable(ticket.getByTestId("panel-collapse"))).toBe(true);
  expect(await reachable(ticket.getByTestId("panel-reset"))).toBe(true);
  // collapsed there, it is still on screen and in reach
  await ticket.getByTestId("panel-collapse").click();
  const c = await settled(ticket);
  for (const ch of await chromeBoxes(page)) expect(overlap(c, ch)).toBe(false);
  expect(await reachable(ticket.getByTestId("panel-collapse"))).toBe(true);
  await ticket.getByTestId("panel-collapse").click();
  // and its own Reset puts it back
  await ticket.getByTestId("panel-reset").click();
  await expect(ticket).not.toHaveAttribute("data-panel-moved", "");
  expect(errors).toEqual([]);
  await ctx.close();
});

test("Today: the resize corner keeps the top-left corner still, and a narrow ticket never cuts its big date", async ({ browser }) => {
  const { ctx, page } = await coastContext(browser);
  await page.goto(`${OK}/today?place=${MOSS}`);
  const ticket = page.locator('[data-panel="ticket"]');
  const t0 = await settled(ticket);
  await drag(page, ticket.locator("[data-panel-resize]"), 60, -80);
  const t1 = await settled(ticket);
  expect(Math.abs(t1.x - t0.x)).toBeLessThanOrEqual(1);
  expect(Math.abs(t1.y - t0.y)).toBeLessThanOrEqual(1);
  expect(Math.round(t1.width - t0.width)).toBe(60);
  expect(Math.round(t0.height - t1.height)).toBe(80);
  // shorter: the content's last part is inside the box or reached by scrolling the ticket's own scroller
  await drag(page, ticket.locator("[data-panel-resize]"), 0, -(t1.height - 420));
  const short = await settled(ticket);
  // it stops at the height of what does not scroll inside (the heading, the action): never shorter than that
  expect(short.height).toBeLessThanOrEqual(t1.height + 1);
  const reach = await ticket.evaluate((n) => {
    const box = n.getBoundingClientRect();
    const last = n.lastElementChild as HTMLElement;
    let deepest: Element = last;
    while (deepest.lastElementChild) deepest = deepest.lastElementChild;
    const lastIn = last.getBoundingClientRect().bottom <= box.bottom + 1;
    // scroll whichever ancestor inside the ticket scrolls, to the end
    for (let e: HTMLElement | null = deepest.parentElement; e && e !== n; e = e.parentElement) if (e.scrollHeight > e.clientHeight + 1) e.scrollTop = e.scrollHeight;
    const d = deepest.getBoundingClientRect();
    return { lastIn, deepIn: d.bottom <= box.bottom + 1 && d.top >= box.top - 1, overflow: n.scrollHeight - n.clientHeight };
  });
  expect(reach.lastIn, "the ticket's last part ends inside its box").toBe(true);
  expect(reach.deepIn, "its last line is reached by scrolling").toBe(true);
  expect(reach.overflow).toBeLessThanOrEqual(1);
  const grip = ticket.locator("[data-panel-grip]");
  const gb = (await grip.boundingBox())!;
  expect(Math.abs(gb.y - short.y)).toBeLessThanOrEqual(2);
  // as narrow as it will go: nothing inside is wider than the ticket
  await drag(page, ticket.locator("[data-panel-resize]"), -400, 0);
  const t2 = await settled(ticket);
  expect(Math.abs(t2.x - t0.x)).toBeLessThanOrEqual(1);
  const cut = await ticket.evaluate((n) => [...n.children].some((c) => c instanceof HTMLElement && c.offsetParent !== null && c.scrollWidth > c.clientWidth + 1));
  expect(cut, "no child draws past the ticket").toBe(false);
  expect(await ticket.evaluate((n) => n.scrollHeight <= n.clientHeight + 1), "nothing draws past its foot").toBe(true);
  await ctx.close();
});

test("a panel enlarged at 1280 keeps the reader's size after a shrink to 1100 and a re-grow", async ({ browser }) => {
  const { ctx, page } = await coastContext(browser, { width: 1280, height: 800 });
  await page.goto(`${OK}/coast?place=${MOSS}`);
  const sea = page.locator('[data-panel="sea-window"]');
  const s0 = await settled(sea);
  await drag(page, sea.locator("[data-panel-grip]"), 0, -200);
  await drag(page, sea.locator("[data-panel-resize]"), 300, 60);
  const big = await settled(sea);
  expect(Math.round(big.width - s0.width)).toBe(300);
  await page.setViewportSize({ width: 1100, height: 600 });
  await page.waitForTimeout(500);
  const small = await settled(sea);
  expect(small.x + small.width).toBeLessThanOrEqual(1100);
  expect(small.y + small.height).toBeLessThanOrEqual(600);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.waitForTimeout(500);
  const back = await settled(sea);
  expect(Math.round(back.width)).toBe(Math.round(big.width));
  expect(Math.round(back.height)).toBe(Math.round(big.height));
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem("nx-panels-v1:coast") ?? "{}").panels?.["sea-window"]);
  expect(saved.w).toBe(Math.round(big.width));
  await ctx.close();
});

for (const [w, h] of [[1100, 760], [900, 700]]) {
  test(`Today: after the screen shrinks to ${w} px a moved ticket finds a free place (no pile-up on the sea panel)`, async ({ browser }) => {
    const { ctx, page } = await coastContext(browser);
    await page.goto(`${OK}/today?place=${MOSS}`);
    const ticket = page.locator('[data-panel="ticket"]');
    const sea = page.locator('[data-panel="sea-next-four-days"]');
    const t0 = await settled(ticket);
    await settled(sea);
    // left, into the middle of the chart, above the sea panel
    await drag(page, ticket.locator("[data-panel-grip]"), -(t0.x - 300), 0);
    await page.setViewportSize({ width: w, height: h });
    await page.waitForTimeout(400);
    const [t, s] = [await settled(ticket), await settled(sea)];
    expect(overlap(t, s), "the ticket and the sea panel do not overlap").toBe(false);
    for (const c of await chromeBoxes(page)) expect(overlap(t, c)).toBe(false);
    expect(t.x + t.width).toBeLessThanOrEqual(w);
    expect(t.y + t.height).toBeLessThanOrEqual(h);
    expect(await ticket.evaluate((n) => n.scrollHeight <= n.clientHeight + 1), "its content scrolls inside it").toBe(true);
    const entry = page.getByTestId("nx-today-explore");
    if (await entry.isVisible()) {
      await entry.scrollIntoViewIfNeeded();
      expect(await reachable(entry), "Explore California Waters is not covered").toBe(true);
    }
    await ctx.close();
  });
}

for (const [path, key] of [["/today", "ticket"], ["/coast", "ticket"]]) {
  test(`${path}: the resting ticket fits inside a 1440×900 screen (its content scrolls)`, async ({ browser }) => {
    const { ctx, page } = await coastContext(browser);
    await page.goto(`${OK}${path}?place=${MOSS}`);
    const t = await settled(page.locator(`[data-panel="${key}"]`));
    expect(t.y + t.height).toBeLessThanOrEqual(900);
    expect(await page.locator(`[data-panel="${key}"]`).evaluate((n) => n.scrollHeight <= n.clientHeight + 1), "its content scrolls inside it").toBe(true);
    await ctx.close();
  });
}

for (const width of [390, 320]) {
  test(`phone ${width} px: no floating panels; the sheet reads in order with nothing overlapping`, async ({ browser }) => {
    const { ctx, page } = await coastContext(browser, { width, height: width === 320 ? 640 : 844 });
    await page.goto(`${OK}/coast?place=${MOSS}`);
    const ticket = page.locator("[data-nx-ticket]");
    const sea = page.locator("[data-nx-dock]");
    await expect(ticket).toBeVisible({ timeout: 20_000 });
    await expect(page.locator("[data-panel-grip]")).toHaveCount(0);
    await expect(page.locator("[data-panel]")).toHaveCount(0);
    const save = page.getByTestId("coast-save-trip");
    await save.scrollIntoViewIfNeeded();
    await expect(save).toBeInViewport();
    const [t, s] = [await ticket.boundingBox(), await sea.boundingBox()];
    expect(overlap(t!, s!), "ticket and sea window stack").toBe(false);
    expect(s!.y).toBeGreaterThanOrEqual(t!.y + t!.height - 1);
    // the primary action is not under the tab bar
    const bar = page.locator("nav").filter({ hasText: "My Coast" }).last();
    if (await bar.isVisible()) expect(overlap((await save.boundingBox())!, (await bar.boundingBox())!)).toBe(false);
    await ctx.close();
  });
}
