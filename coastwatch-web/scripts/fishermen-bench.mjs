// Six fixed task walkthroughs (docs/coastwatch/fishermen-first/07-benchmarks.md), scripted in a
// phone-sized browser against a CoastWatch server running the committed fixture data with replayed
// ocean conditions (no live third-party requests). These are scripted simulations, not user tests.
//
//   node scripts/fishermen-bench.mjs <base-url> <driver: before|after> <out.json>
//
// Counted per task: success against the written rubric, taps, typed characters, scrolls needed to
// bring the answer on screen, distinct pages, sources outside Neritic still needed, and the text
// a person would read first (for the misreading review). Time is scripted speed, not human time.
import { writeFileSync } from "node:fs";
import { chromium } from "@playwright/test";

const [base, driver, out] = process.argv.slice(2);
if (!base || !driver || !out) throw new Error("usage: fishermen-bench.mjs <base-url> <before|after> <out.json>");

/** A previous visit 7 days earlier at Crescent City (commercial, Dungeness), for T5. */
async function seedReturn(page) {
  const r = await page.request.get(`${base}/api/briefing?place=port-201&for=commercial&species=dungeness`);
  if (!r.ok()) return false; // the before build has no change briefing
  const snap = await r.json();
  snap.official.statements.find((s) => s.id === "cdfw-dungeness-2026-27-season-status").statement = "Commercial Fishery: Season status to be announced.";
  snap.official.notices = snap.official.notices.filter((n) => !/razor-clam-del-norte|annual-mussel-quarantine/.test(n.id));
  const entry = { key: "port-201|commercial|dungeness", place: "port-201", placeName: "Crescent City", audience: "commercial", species: "dungeness", speciesLabel: "Dungeness crab", lastSeenAt: "2026-10-01T16:00:00Z", snapshot: snap };
  await page.goto(`${base}/about`);
  await page.evaluate((v) => localStorage.setItem("cw-waters-v1", JSON.stringify([v])), entry);
  return true;
}

function counter(page) {
  const c = { taps: 0, typed: 0, scrolls: 0, pages: new Set() };
  page.on("framenavigated", (f) => {
    if (f === page.mainFrame()) {
      const u = new URL(f.url());
      if (u.pathname !== "/about") c.pages.add(u.pathname);
    }
  });
  return {
    c,
    tap: async (loc) => {
      await loc.first().scrollIntoViewIfNeeded();
      await loc.first().tap();
      c.taps++;
      await page.waitForTimeout(500);
    },
    type: async (loc, text) => {
      await loc.fill(text);
      c.typed += text.length;
      await page.waitForTimeout(600);
    },
  };
}

/** Bring a visible element on screen; a scroll counts when it was below the first screen. */
async function onScreen(page, loc, c) {
  const el = loc.first();
  await el.waitFor({ state: "visible", timeout: 12000 });
  const box = await el.boundingBox();
  const vh = page.viewportSize().height;
  if (box && (box.y > vh - 40 || box.y < 0)) {
    await el.scrollIntoViewIfNeeded();
    c.scrolls++;
  }
  return el;
}

const visible = async (loc) => (await loc.count()) > 0 && (await loc.first().isVisible());

// ---------------------------------------------------------------- tasks
const TASKS = {
  async T1(page, k, d) {
    // A Monterey commercial fisherman checks tomorrow's conditions (cold first visit, front door)
    await page.goto(`${base}/`);
    await k.tap(page.getByTestId("welcome-fish"));
    await k.tap(page.getByTestId("intro-purpose-commercial"));
    await k.tap(page.getByTestId("place-input"));
    await k.type(page.getByTestId("place-input"), "Monterey");
    await k.tap(page.locator("[role=option]").filter({ hasText: /^Monterey/ }));
    await page.waitForURL(/\/coast\?/);
    if (d === "before" && !(await visible(page.getByTestId("conditions")))) await k.tap(page.getByTestId("chapter-head-conditions"));
    const el = await onScreen(page, page.getByTestId("conditions"), k.c);
    const text = await el.innerText();
    const periods = (text.match(/\b(Today|Tonight|Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*(day| night)?\b/g) ?? []).length;
    return { ok: periods >= 2 && /issued|Issued/.test(text), answer: text.slice(0, 400), external: 0 };
  },
  async T2(page, k, d) {
    // A Crescent City Dungeness crab fisherman checks notices and applicable restrictions
    await page.goto(`${base}/coast?place=port-201`);
    await k.tap(page.getByTestId("coast-for-commercial"));
    await k.tap(page.getByTestId("coast-species-dungeness"));
    const first = (await page.getByTestId("coast-answer").innerText()).split("\n")[0];
    const season = page.getByText(/Season is closed/).filter({ visible: true });
    if (!(await visible(season)) && d === "before") await k.tap(page.getByTestId("chapter-head-before"));
    const el = await onScreen(page, page.getByText(/Season is closed/).filter({ visible: true }), k.c);
    const text = await el.innerText();
    return { ok: /Season is closed/.test(text), answer: text.slice(0, 300), firstLine: first, external: 0 };
  },
  async T3(page, k, d) {
    // A Clam Beach harvester checks the latest official shellfish information
    await page.goto(`${base}/coast?place=site-clam-beach`);
    await k.tap(page.getByTestId("coast-for-shellfish"));
    await k.tap(page.getByTestId("coast-species-razor-clam"));
    if (d === "before" && !(await visible(page.getByTestId("brief-official")))) await k.tap(page.getByTestId("journey-next"));
    const el = await onScreen(page, page.getByTestId("brief-official"), k.c);
    const text = await el.innerText();
    return { ok: /razor clam/i.test(text) && /Since|dated|20\d\d/.test(text) && (await el.getByRole("link").count()) > 0, answer: text.slice(0, 400), external: 0 };
  },
  async T4(page, k) {
    // A Santa Cruz fisherman investigates recent HAB activity
    await page.goto(`${base}/coast?place=port-593`);
    const el = await onScreen(page, page.getByTestId("coast-answer"), k.c);
    const text = await el.innerText();
    // a dated nearby sample (toxin or cells), with its station: either wording passes
    return { ok: /(Wharf|Pier|station).*\b(\d+ days? ago|today|yesterday)\b/i.test(text), answer: text.slice(0, 300), external: 0 };
  },
  async T5(page, k, d) {
    // A fisherman returns after several days and wants to see what changed
    const seeded = await seedReturn(page);
    k.c.pages.clear();
    await page.goto(`${base}/coast?place=port-201`);
    if (!seeded || d === "before") {
      await page.waitForTimeout(2500);
      // nothing in the app lists changes: the person compares the agency pages themselves
      return { ok: false, answer: "No list of changes. The person must re-read CDFW health advisories, CDPH releases and the Whale Safe page and compare from memory.", external: 3 };
    }
    await page.getByTestId("since-last-check").waitFor({ state: "visible", timeout: 12000 });
    const resume = page.getByTestId("since-resume");
    if (await visible(resume)) await k.tap(resume);
    const el = await onScreen(page, page.getByTestId("since-list"), k.c);
    const text = await el.innerText();
    return { ok: /now says/.test(text) && /last check/.test(text), answer: text.slice(0, 500), external: 0 };
  },
  async T6(page, k) {
    // A researcher examines the same observations and their provenance
    await page.goto(`${base}/coast?place=port-550&mode=research`);
    const meas = page.getByTestId("coast-measured");
    if (!(await visible(meas)) && (await visible(page.getByTestId("coast-details")))) await k.tap(page.getByTestId("coast-details"));
    const el = await onScreen(page, meas, k.c);
    const text = await el.innerText();
    const links = await el.getByRole("link").evaluateAll((as) => as.map((a) => a.href));
    return { ok: /20\d\d|ago/.test(text) && links.some((h) => /erddap|sccoos|calhabmap/i.test(h)), answer: text.slice(0, 300), external: 0 };
  },
};

const browser = await chromium.launch();
const results = [];
for (const [id, run] of Object.entries(TASKS)) {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  const k = counter(page);
  const t0 = Date.now();
  let r;
  try {
    r = await run(page, k, driver);
  } catch (e) {
    r = { ok: false, answer: `ERROR: ${e.message.split("\n")[0]}`, external: null };
  }
  const ms = Date.now() - t0;
  await page.screenshot({ path: out.replace(/\.json$/, `-${id}.png`) });
  results.push({ id, driver, ok: r.ok, ms, taps: k.c.taps, typed: k.c.typed, scrolls: k.c.scrolls, pages: k.c.pages.size, external: r.external, firstLine: r.firstLine ?? null, answer: r.answer });
  await ctx.close();
}
await browser.close();
writeFileSync(out, JSON.stringify(results, null, 1));
for (const r of results) console.log(`${r.id} ${r.driver} ok=${r.ok} ${r.ms}ms taps=${r.taps} typed=${r.typed} scrolls=${r.scrolls} pages=${r.pages} external=${r.external}`);
