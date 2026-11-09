// The demo's fallback screenshots (docs/coastwatch/pilot/pitch/demo-script.md, "No-wifi fallback"):
// every step of the script as a PNG, phone width in the dark and daylight themes, plus desktop.
//
//   node scripts/pilot-shots.mjs <base-url> [out-dir] [place]
//   e.g. node scripts/pilot-shots.mjs http://localhost:3331 ../docs/coastwatch/pilot/pitch/shots port-592
//
// It drives a running server (it starts nothing and writes nothing but the PNGs), so the pictures
// show whatever data that server reads: point it at the pilot build the night before the demo.
// Ask Neritic is asked two questions per theme (the server's rate limit is 30 per 10 minutes).
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const BASE = (process.argv[2] ?? "").replace(/\/$/, "");
if (!BASE) throw new Error("usage: node scripts/pilot-shots.mjs <base-url> [out-dir] [place]");
const OUT = path.resolve(process.argv[3] ?? "pilot-shots");
const PLACE = process.argv[4] ?? "port-592";
const TODAY = `/today?place=${PLACE}`;
// the phone dock is fixed to the bottom edge: for a picture of one section it is taken away, band and all
const NO_DOCK = 'div:has(> [data-testid="nav-dock"]){display:none !important}';
const QUESTION = "Is the Monterey crab season open?";
const REFUSAL = "Is it safe to eat the crab?";

const THEMES = [
  ["phone-dark", { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 }, false],
  ["phone-day", { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 }, true],
  ["desktop-dark", { viewport: { width: 1440, height: 900 } }, false],
];

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch({ channel: process.env.CI ? undefined : "chrome" });
const done = [];

for (const [theme, options, day] of THEMES) {
  const ctx = await browser.newContext(options);
  await ctx.addInitScript((d) => {
    try {
      localStorage.setItem("cw-intro-dismissed-v1", "1");
      if (d) localStorage.setItem("nx-daylight-v1", "1");
    } catch {}
  }, day);
  const page = await ctx.newPage();
  const shot = async (name, target = null, opts = {}) => {
    const file = path.join(OUT, `${name}-${theme}.png`);
    if (target) await target.screenshot({ path: file, ...opts });
    else await page.screenshot({ path: file, ...opts });
    done.push(path.basename(file));
  };
  const today = async () => {
    await page.goto(BASE + TODAY);
    await page.getByTestId("today-port").waitFor();
    // the sea arrives after the page (NOAA); a server that cannot reach it shows the plain error instead
    await page.getByTestId("sea-lanes").or(page.getByTestId("sea-error")).first().waitFor({ timeout: 30_000 }).catch(() => undefined);
    await page.waitForTimeout(700);
  };
  // a section on its own, without the dock across it
  const section = async (name, testid) => {
    const dock = await page.addStyleTag({ content: NO_DOCK });
    const el = page.getByTestId(testid);
    await el.scrollIntoViewIfNeeded();
    await page.waitForTimeout(250);
    await shot(name, el);
    await dock.evaluate((n) => n.remove());
  };

  // 01 Today: the port, the answer, the status strip
  await today();
  await shot("01-today-top");
  if (theme === "desktop-dark") await shot("01-today-full", null, { fullPage: true });
  await section("02-status-strip", "today-status");
  // a status row's detail
  const row = page.locator('[data-testid^="status-"][aria-haspopup="dialog"]').first();
  if (await row.count()) {
    await row.click();
    await page.getByTestId("status-sheet").waitFor();
    await page.waitForTimeout(500);
    await shot("02-status-detail");
    await page.keyboard.press("Escape");
  }
  await section("03-toxin-tests", "today-tests");
  await section("04-sea-weather", "today-sea");
  await section("05-bloom-outlook", "today-bloom");
  await section("06-whale-rules", "today-whale");

  // 07 what changed: this visit is the first on this device; the second shows what differs. So that the
  // picture shows the list itself, the remembered visit is made a week old and one fact older (as the e2e does).
  await page.evaluate((place) => {
    const k = `nx-today-seen-v1:${place}`;
    const s = JSON.parse(localStorage.getItem(k) ?? "null");
    if (!s) return;
    s.at = new Date(Date.parse(s.at) - 7 * 86_400_000).toISOString();
    const whale = s.facts.find((f) => f.kind === "whale");
    if (whale) whale.headline = "Whale rules: no action on record.";
    s.facts = s.facts.filter((f) => !f.id.startsWith("test:") || f !== s.facts.find((x) => x.id.startsWith("test:")));
    localStorage.setItem(k, JSON.stringify(s));
  }, PLACE);
  await today();
  await section("07-what-changed", "today-changed");

  // 08 save for offline
  await page.getByTestId("save-offline").scrollIntoViewIfNeeded();
  await page.getByTestId("save-offline").click();
  await page.getByTestId("save-status").filter({ hasText: /Saved|Could not/ }).waitFor({ timeout: 45_000 }).catch(() => undefined);
  await section("08-save-offline", "today-actions");

  // 09 alert setup: from Today's "Set an alert"
  await page.getByTestId("action-alert").click();
  await page.getByTestId("account-page").waitFor();
  await page.waitForTimeout(600);
  await shot("09-alert-arrive");
  const save = page.getByTestId("want-save");
  if (await save.count()) {
    await save.click();
    await page.getByTestId(`alerts-${PLACE}`).waitFor();
    await page.waitForTimeout(500);
    await section("09-alert-setup", "account-alerts");
  }

  // 10 Ask Neritic: a cited answer, then the refusal to call seafood safe
  await page.goto(`${BASE}/ask?place=${PLACE}`);
  await page.getByTestId("ask-input").waitFor();
  await shot("10-ask-empty");
  for (const [name, q] of [["10-ask-cited-answer", QUESTION], ["11-ask-safe-to-eat-refusal", REFUSAL]]) {
    const before = await page.getByTestId("ask-answer").count();
    await page.getByTestId("ask-input").fill(q);
    await page.getByTestId("ask-send").click();
    await page.getByTestId("ask-answer").nth(before).waitFor({ timeout: 60_000 });
    const turn = page.getByTestId("ask-turns").locator("> li").last();
    if (name.startsWith("10")) await turn.getByTestId("ask-chip").first().click();
    await page.waitForTimeout(300);
    const dock = await page.addStyleTag({ content: NO_DOCK });
    await turn.scrollIntoViewIfNeeded();
    await shot(name, turn);
    await dock.evaluate((n) => n.remove());
  }
  // the same chat, in place on Today
  await today();
  await page.getByTestId("hero-ask").click();
  await page.getByTestId("ask-sheet").waitFor();
  await page.waitForTimeout(700);
  await shot("10-ask-sheet-on-today");
  await page.keyboard.press("Escape");

  // 12 Explore and 13 Research: one tap away
  await page.goto(`${BASE}/?view=map&sel=harbor:${PLACE.replace(/^port-/, "")}`);
  await page.getByTestId("explore").waitFor();
  await page.waitForTimeout(6000); // the basemap and the field settle
  await shot("12-explore-map");
  await page.goto(`${BASE}/research?place=${PLACE}`);
  await page.waitForTimeout(3500);
  await shot("13-research");
  await ctx.close();
}
await browser.close();
console.log(`${done.length} screenshots in ${OUT}\n${done.join("\n")}`);
