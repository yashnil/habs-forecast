// Eight scripted fishing journeys on a phone (390 x 844), run against a CoastWatch server with the
// committed fixture data, a fixed clock and replayed NOAA responses (no live third-party requests).
// The same rubric is applied to the build before the trip briefing ("before") and after ("after").
// These are scripted simulations of the tasks, not tests with people.
//
//   node scripts/trip-bench.mjs <base-url> <before|after> <out.json>
//
// For each journey and each rubric item: found or not (a dead end), on the first screen or not,
// how far a person must scroll to reach it (screens), taps needed, and the scripted time until every
// item is on the page. "Time" is the browser's speed, not a person's reading time.
import { writeFileSync } from "node:fs";
import { chromium } from "@playwright/test";

const [base, driver, out] = process.argv.slice(2);
if (!base || !driver || !out) throw new Error("usage: trip-bench.mjs <base-url> <before|after> <out.json>");
const NOW = "2026-10-08T20:00:00Z";
const VH = 844;

const by = (page, id) => page.getByTestId(id);
const any = (page, ...ids) => page.locator(ids.map((i) => `[data-testid="${i}"]`).join(", "));

/** Each journey: the link a person opens, then what they must find (per driver), with any taps needed. */
const J = {
  "J1 Crescent City, commercial Dungeness crab": {
    url: "/coast?place=port-201&for=commercial&species=dungeness",
    find: {
      after: [
        ["season status, quoted, with CDFW's page", (p) => by(p, "matter-season")],
        ["HAB evidence near the waters", (p) => by(p, "matter-hab")],
        ["sea conditions", (p) => by(p, "matter-conditions")],
        ["CDFW closure line to call", (p) => by(p, "check-closure-line")],
        ["what to check before leaving (list)", (p) => by(p, "checklist")],
      ],
      before: [
        ["season status, quoted, with CDFW's page", (p) => by(p, "answer-status")],
        ["HAB evidence near the waters", (p) => by(p, "answer-measured")],
        ["sea conditions", (p) => by(p, "conditions")],
        ["CDFW closure line to call", (p) => by(p, "step-closure-line"), ["next-steps-more"]],
        ["what to check before leaving (list)", (p) => by(p, "next-steps")],
      ],
    },
  },
  "J2 Monterey, commercial anchovy": {
    url: "/coast?place=port-550&for=commercial&species=anchovy-sardine",
    find: {
      after: [
        ["the notices that cover these waters", (p) => by(p, "matter-notices")],
        ["the latest HAB sample nearby, with distance and age", (p) => by(p, "matter-hab")],
        ["the agency source of the notice", (p) => by(p, "matter-notices-source")],
        ["first thing to check", (p) => by(p, "checklist").locator("li").first()],
      ],
      before: [
        ["the notices that cover these waters", (p) => by(p, "answer-official")],
        ["the latest HAB sample nearby, with distance and age", (p) => by(p, "answer-measured")],
        ["the agency source of the notice", (p) => by(p, "brief-official").getByRole("link").first()],
        ["first thing to check", (p) => by(p, "next-steps").locator('[data-testid^="step-"]').first()],
      ],
    },
  },
  "J3 Santa Cruz, recreational fishing (fresh HAB data)": {
    url: "/coast?place=port-593&for=recreational",
    find: {
      after: [
        ["the notices that cover these waters", (p) => by(p, "matter-notices")],
        ["the latest HAB sample nearby", (p) => by(p, "matter-hab")],
        ["HAB samples over time (visual)", (p) => by(p, "evidence-timeline")],
        ["the region's sport fishing rules", (p) => p.locator('[data-testid^="check-region-"]'), ["checklist-more?"]],
      ],
      before: [
        ["the notices that cover these waters", (p) => by(p, "answer-official")],
        ["the latest HAB sample nearby", (p) => by(p, "answer-measured")],
        ["HAB samples over time (visual)", (p) => p.locator('[data-testid^="ev-cells"], [data-testid^="ev-toxin"]')],
        ["the region's sport fishing rules", (p) => p.locator('[data-testid^="step-region-"]'), ["next-steps-more?"]],
      ],
    },
  },
  "J4 Clam Beach, recreational razor clam harvest": {
    url: "/coast?place=site-clam-beach&for=shellfish&species=razor-clam",
    find: {
      after: [
        ["the razor clam closures and health advisory", (p) => by(p, "matter-notices")],
        ["CDPH's shellfish advisory map and testing", (p) => by(p, "check-cdph-map")],
        ["the lowest tide of the day", (p) => by(p, "matter-conditions")],
      ],
      before: [
        ["the razor clam closures and health advisory", (p) => by(p, "answer-official")],
        ["CDPH's shellfish advisory map and testing", (p) => by(p, "step-cdph-map"), ["next-steps-more?"]],
        ["the lowest tide of the day", (p) => by(p, "cond-tides")],
      ],
    },
  },
  "J5 Shelter Cove, sparse monitoring": {
    url: "/coast?place=port-215&for=commercial",
    find: {
      after: [
        ["that no recent sample describes these waters", (p) => by(p, "matter-hab")],
        ["where the nearest stations are, and how far", (p) => by(p, "evidence-where")],
        ["where to look instead (CDPH results)", (p) => by(p, "check-toxin-gap"), ["checklist-more?"]],
      ],
      before: [
        ["that no recent sample describes these waters", (p) => by(p, "answer-measured")],
        ["where the nearest stations are, and how far", (p) => by(p, "hab-summary")],
        ["where to look instead (CDPH results)", (p) => by(p, "step-toxin-gap"), ["next-steps-more?"]],
      ],
    },
  },
  "J6 San Diego, outside the toxin model's values": {
    url: "/coast?place=port-880&for=recreational",
    find: {
      after: [
        ["that the model has no value for these waters", (p) => by(p, "matter-hab")],
        ["the model row, marked as not covering the waters", (p) => by(p, "ev-label-model")],
      ],
      before: [
        ["that the model has no value for these waters", (p) => by(p, "answer-model")],
        ["the model row, marked as not covering the waters", (p) => by(p, "model-where")],
      ],
    },
  },
  "J7 Crescent City, a trip in three days": {
    url: "/coast?place=port-201&for=commercial&species=dungeness",
    pick: { after: ["trip-day", "@last-day"], before: ["coast-day:last"] },
    find: {
      after: [
        ["the chosen day is shown", (p) => by(p, "trip-day")],
        ["that day's tides", (p) => by(p, "matter-conditions")],
      ],
      before: [
        ["the chosen day is shown", (p) => by(p, "coast-day")],
        ["that day's tides", (p) => by(p, "cond-tides")],
      ],
    },
  },
  "J8 A returning fisherman checks what changed": {
    url: "/coast?place=port-201&for=commercial&species=dungeness",
    seed: true,
    find: {
      after: [["the list of changes since the last visit", (p) => by(p, "since-list")]],
      before: [["the list of changes since the last visit", (p) => by(p, "since-list")]],
    },
  },
};

/** A previous visit seven days earlier at Crescent City (commercial Dungeness), with two differences. */
async function seed(page) {
  const r = await page.request.get(`${base}/api/briefing?place=port-201&for=commercial&species=dungeness`);
  if (!r.ok()) return;
  const snap = await r.json();
  const st = snap.official.statements.find((s) => /season-status/.test(s.id));
  if (st) st.statement = "Commercial Fishery: Season status to be announced.";
  const entry = { key: "port-201|commercial|dungeness", place: "port-201", placeName: "Crescent City", audience: "commercial", species: "dungeness", speciesLabel: "Dungeness crab", lastSeenAt: "2026-10-01T16:00:00Z", snapshot: snap };
  await page.goto(`${base}/about`);
  await page.evaluate((v) => localStorage.setItem("cw-waters-v1", JSON.stringify([v])), entry);
}

async function run(browser, name, j) {
  const ctx = await browser.newContext({ viewport: { width: 390, height: VH }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
  await ctx.addInitScript(() => {
    try {
      localStorage.setItem("cw-intro-dismissed-v1", "1");
    } catch {}
  });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  if (j.seed) await seed(page);
  let taps = 0;
  const t0 = Date.now();
  await page.goto(base + j.url);
  // choosing the day
  for (const a of j.pick?.[driver] ?? []) {
    if (a === "@last-day") {
      const opts = page.getByTestId("trip-sheet-day").locator('[role="radio"]');
      await opts.last().waitFor({ timeout: 10000 });
      await opts.last().tap();
      taps++;
    } else if (a.endsWith(":last")) {
      const sel = page.getByTestId(a.split(":")[0]);
      await sel.waitFor({ timeout: 10000 });
      const vals = await sel.locator("option").evaluateAll((os) => os.map((o) => o.value));
      await sel.selectOption(vals.at(-1));
      taps += 2;
    } else {
      await page.getByTestId(a).tap();
      taps++;
    }
    await page.waitForTimeout(400);
  }
  const items = [];
  for (const [what, loc, pre] of j.find[driver]) {
    let l = loc(page);
    let found = false;
    try {
      await l.first().waitFor({ state: "attached", timeout: 9000 });
      found = true;
    } catch {
      // perhaps behind a "more" control
      for (const p of pre ?? []) {
        const id = p.replace(/\?$/, "");
        const b = page.getByTestId(id);
        if ((await b.count()) && (await b.first().isVisible())) {
          await b.first().scrollIntoViewIfNeeded();
          await b.first().tap();
          taps++;
          await page.waitForTimeout(400);
        }
      }
      l = loc(page);
      try {
        await l.first().waitFor({ state: "attached", timeout: 3000 });
        found = true;
      } catch {
        found = false;
      }
    }
    let firstScreen = false;
    let screens = null;
    let text = "";
    if (found) {
      const visible = await l.first().isVisible();
      const y = await l.first().evaluate((e) => e.getBoundingClientRect().top + window.scrollY);
      const h = await l.first().evaluate((e) => Math.min(e.getBoundingClientRect().height, 160));
      firstScreen = visible && y + Math.min(h, 60) <= VH - 64; // above the tab bar
      screens = Math.max(0, Math.round(((y + Math.min(h, 60) - (VH - 64)) / VH) * 10) / 10);
      text = (await l.first().innerText().catch(() => "")).replace(/\s+/g, " ").slice(0, 160);
      if (!visible) found = false;
    }
    items.push({ what, found, firstScreen, screens, text });
  }
  const ms = Date.now() - t0;
  const pageH = await page.evaluate(() => document.documentElement.scrollHeight);
  await ctx.close();
  return { name, items, taps, ms, pageScreens: Math.round((pageH / VH) * 10) / 10 };
}

const browser = await chromium.launch();
const results = [];
for (const [name, j] of Object.entries(J)) results.push(await run(browser, name, j));
await browser.close();
writeFileSync(out, JSON.stringify({ driver, base, at: new Date().toISOString(), results }, null, 2));
for (const r of results) {
  const f = r.items.filter((i) => i.found).length;
  const fs = r.items.filter((i) => i.firstScreen).length;
  console.log(`${driver.padEnd(6)} ${r.name.padEnd(52)} found ${f}/${r.items.length}  first screen ${fs}/${r.items.length}  taps ${r.taps}  scroll ${r.items.reduce((a, i) => a + (i.screens ?? 0), 0).toFixed(1)} screens  page ${r.pageScreens} screens  ${r.ms} ms`);
}
