#!/usr/bin/env node
/**
 * v14 actual stills: the APP at the same 25 states as the prototype references (scripts/v14-reference-shots.mjs),
 * then one side-by-side composite per state (reference | actual) and compare.md listing them.
 *
 *   trip-empty  /coast?place=port-592&view=trip, catch 400 and price 9 typed, fuel and crew left blank
 *   trip-done   the regression trip: 400 lb at $9, 30 gal at $8.24, bait and ice $100, fees 0 %, 1 crew at 20 % after
 *               costs, fuel and bait/ice shared (off the top). Net to boat $2,602.24
 *   today       /today?place=port-223 (Fort Bragg), a returning reader whose home port is Fort Bragg
 *   research    /research?station=HABs-SantaCruzWharf&var=pDA&range=3y&day=2026-05-27 (the 3-year range, the May 27,
 *               2026 sample selected; on a phone the sample's details are opened if the page keeps them closed)
 *   alerts      /alerts?place=port-223 with Fort Bragg followed and one controlled change: the published records
 *               are served twice, the second time with the commercial Dungeness status moved to Jan 15, 2027 (the
 *               prototype's example), exactly as tests/e2e/v13/alerts.spec.ts injects a change
 *
 * Live values differ from the prototype's examples: the comparison is about composition, type, colour, cartography
 * and hierarchy. Every state is set up through roles, labels and test ids (see SELECTORS below); a step that fails
 * is written to notes.json and the still is taken anyway, so one broken selector never hides the other 24.
 *
 * Usage (from coastwatch-web, after `npm run data:fixture && npm run build`):
 *   node scripts/v14-app-shots.mjs                      starts `next start` on CW_E2E_PORT_BASE+10 (default 5910)
 *                                                       on fixture data (fixture clock 2026-10-13, recorded NOAA)
 *   CW_DATA_BASE_URL=https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data-staging/v1 \
 *     node scripts/v14-app-shots.mjs                    the same on staging data (live clock and conditions)
 *   node scripts/v14-app-shots.mjs --base http://localhost:5900    an app that is already running
 *   env: FRAMES=today,alerts  SIZES=390x844,320x640  OUT=<dir> (default docs/coastwatch/v14/implementation)
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright";
import { IMPL_DIR, PHONE_MAX, contextFor, imageTools, pickFrames, pickSizes, sizeName } from "./v14-lib.mjs";

const arg = (k) => {
  const i = process.argv.indexOf(k);
  return i > 0 ? process.argv[i + 1] : undefined;
};
const OUT = process.env.OUT ?? IMPL_DIR;
const DATA = process.env.CW_DATA_BASE_URL ?? "/data/fixture/v1";
const FIXTURE = !/^https?:/.test(DATA);
const PORT = Number(process.env.CW_E2E_PORT_BASE ?? 5900) + 10;
let BASE = arg("--base");
const SETTLE = Number(process.env.SETTLE ?? 2500);

const TRIP_PORT = "port-592"; // Moss Landing
const FORT_BRAGG = "port-223";
const RESEARCH = { station: "HABs-SantaCruzWharf", v: "pDA", range: "3y", day: process.env.CW_V14_RESEARCH_DAY ?? "2026-05-27" };

/** Everything the scenarios touch. Roles and labels first; test ids where a role cannot say it. */
const SELECTORS = {
  tripInputs: "trip-inputs", // testid: the trip's input block (My Coast, B)
  tripNet: ["trip-netbar-value", "trip-net"], // testid: the net-to-boat figure (either form)
  afterCosts: { role: "radio", name: "After costs" },
  payerFuel: { group: "Who pays fuel", radio: "Shared" },
  payerBait: { group: "Who pays bait and ice", radio: "Shared" },
  researchPlot: "research-plot", // testid: the measured plot (Research, A)
  sampleDetails: "sample-details", // testid: phone button that opens the selected sample (Research, A)
  inspector: "research-inspector", // testid
  alertsFollow: "alerts-follow", // testid: Follow <port> (Alerts, E)
  alertsSaid: "alerts-said", // testid: the live status line
  alertsCheck: "alerts-check-now", // testid
  alertsItem: "alerts-item", // testid: one alert in the inbox
};

const store = (home) => ({
  "nx-opening-seen-v1": "1",
  "nx-tour-v1": "dismissed",
  "nx-chart-mode-v1": "day",
  "nx-port-v1": JSON.stringify({ v: 1, place: home, for: "all", species: "any", lastPort: home, fisheries: ["dungeness_commercial"], vessel: null }),
  "nx-coast-v1": JSON.stringify({ fishery: "dungeness_commercial" }),
});

const textbox = (page, label) => page.getByRole("textbox", { name: label, exact: true });
async function type(page, label, v) {
  const f = textbox(page, label);
  await f.scrollIntoViewIfNeeded();
  await f.fill(v);
}
const radioIn = (page, { group, radio }) => page.getByRole("radiogroup", { name: group }).getByRole("radio", { name: radio });

/** Two states of one port's published records (tests/e2e/v13/helpers.ts secondState). */
async function secondState(page, port, edit) {
  let published = false;
  await page.route(`**/today/data?place=${port}`, async (route) => {
    const res = await route.fetch();
    if (!published) return route.fulfill({ response: res });
    const j = await res.json();
    edit(j.inputs);
    j.inputs.dataGeneratedAt = new Date(Date.parse(j.inputs.dataGeneratedAt ?? Date.now()) + 30 * 60_000).toISOString();
    await route.fulfill({ response: res, json: j });
  });
  return () => (published = true);
}
/** The prototype's example change: the commercial Dungeness delay moves from Jan 1 to Jan 15, 2027. */
function dungenessMoves(inputs) {
  const recs = inputs?.fisheries?.records ?? [];
  const rec = recs.find((x) => x.fishery === "dungeness_commercial") ?? recs[0];
  if (!rec) throw new Error("no fisheries record to change");
  rec.headline = "Commercial season delayed to at least Jan 15, 2027.";
}

const SCENARIOS = {
  "trip-empty": {
    home: TRIP_PORT,
    url: `/coast?place=${TRIP_PORT}&view=trip`,
    async setup(page, step) {
      await step("trip inputs visible", () => page.getByTestId(SELECTORS.tripInputs).waitFor({ timeout: 20_000 }));
      await step("type catch and price", async () => {
        await type(page, "Catch", "400");
        await type(page, "Price", "9");
      });
    },
  },
  "trip-done": {
    home: TRIP_PORT,
    url: `/coast?place=${TRIP_PORT}&view=trip`,
    async setup(page, step, note) {
      await step("trip inputs visible", () => page.getByTestId(SELECTORS.tripInputs).waitFor({ timeout: 20_000 }));
      await step("type the regression trip", async () => {
        for (const [l, v] of [["Catch", "400"], ["Price", "9"], ["Fuel", "30"], ["Fuel price", "8.24"], ["Bait and ice", "100"], ["Fees", "0"], ["Crew", "1"]]) await type(page, l, v);
      });
      await step("crew after costs, 20 %", async () => {
        await page.getByRole(SELECTORS.afterCosts.role, { name: SELECTORS.afterCosts.name }).click();
        await type(page, "Crew share", "20");
      });
      await step("fuel and bait/ice shared (off the top)", async () => {
        await radioIn(page, SELECTORS.payerFuel).click();
        await radioIn(page, SELECTORS.payerBait).click();
      });
      await step("net to boat shown", async () => {
        const net = page.locator(SELECTORS.tripNet.map((t) => `[data-testid="${t}"]`).join(",")).first();
        await net.waitFor({ timeout: 5000 });
        await page.waitForTimeout(1200); // rolling digits
        const said = (await net.innerText()).replace(/\s+/g, " ").trim();
        note("net to boat", said);
        if (!said.includes("2,602")) throw new Error(`expected $2,602.24, read "${said}"`);
      });
    },
  },
  today: {
    home: FORT_BRAGG,
    url: `/today?place=${FORT_BRAGG}`,
    async setup(page, step) {
      await step("Today heading", () => page.getByRole("heading", { level: 1 }).first().waitFor({ timeout: 20_000 }));
    },
  },
  research: {
    home: TRIP_PORT,
    url: `/research?station=${RESEARCH.station}&var=${RESEARCH.v}&range=${RESEARCH.range}&day=${RESEARCH.day}`,
    async setup(page, step, note, size) {
      await step("plot drawn", () => page.getByTestId(SELECTORS.researchPlot).waitFor({ timeout: 25_000 }));
      if (size[0] <= PHONE_MAX)
        await step("open the sample on a phone", async () => {
          const b = page.getByTestId(SELECTORS.sampleDetails);
          if (await b.isVisible().catch(() => false)) await b.click();
        });
      await step("the May 27 sample is the one shown", async () => {
        const insp = page.getByTestId(SELECTORS.inspector);
        await insp.waitFor({ timeout: 8000 });
        const t = (await insp.innerText()).replace(/\s+/g, " ");
        note("inspector", t.slice(0, 200));
        const d = new Date(`${RESEARCH.day}T12:00:00Z`);
        const words = d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
        if (!t.includes(words) && !t.includes(RESEARCH.day)) throw new Error(`inspector does not show ${words}`);
      });
    },
  },
  alerts: {
    home: FORT_BRAGG,
    url: `/alerts?place=${FORT_BRAGG}`,
    async prepare(page) {
      return secondState(page, FORT_BRAGG, dungenessMoves);
    },
    async setup(page, step, note, size, publish) {
      await step("follow Fort Bragg", async () => {
        await page.getByTestId(SELECTORS.alertsFollow).click({ timeout: 20_000 });
        await page.getByTestId(SELECTORS.alertsSaid).filter({ hasText: /noted as it is now/ }).waitFor({ timeout: 15_000 });
      });
      publish();
      await step("check now makes one alert", async () => {
        await page.getByTestId(SELECTORS.alertsCheck).click();
        await page.getByTestId(SELECTORS.alertsItem).first().waitFor({ timeout: 15_000 });
        note("alerts", await page.getByTestId(SELECTORS.alertsItem).count());
      });
      // the frame shows the page as a reader finds it: the inbox from the top
      await page.evaluate(() => scrollTo(0, 0));
    },
  },
};

// ---------------------------------------------------------------- server
let server = null;
async function up(url) {
  for (let i = 0; i < 120; i++) {
    try {
      if ((await fetch(`${url}/about`)).ok) return;
    } catch {
      // not yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`no app at ${url}`);
}
if (!BASE) {
  BASE = `http://localhost:${PORT}`;
  const env = { ...process.env, CW_DATA_BASE_URL: DATA };
  if (FIXTURE) Object.assign(env, { CW_CONDITIONS_FIXTURE: "tests/conditions-fixtures.json", CW_NWS_API: "1", CW_FIXED_NOW: "2026-10-13T06:00:00Z" });
  server = spawn("npx", ["next", "start", "-p", String(PORT)], { env, stdio: "ignore", detached: true });
  console.log(`next start on ${PORT} (${FIXTURE ? "fixture" : DATA})`);
}
const stop = () => {
  if (server) {
    try {
      process.kill(-server.pid);
    } catch {
      // gone
    }
  }
};
process.on("exit", stop);
process.on("SIGINT", () => process.exit(130));

// ---------------------------------------------------------------- run
await up(BASE);
for (const d of ["actual", "compare"]) mkdirSync(join(OUT, d), { recursive: true });
const browser = await chromium.launch({ channel: process.env.CW_PROOF_CHANNEL ?? "chrome" });
const tools = await imageTools(browser);
const notes = {};
const rows = [];
try {
  for (const size of pickSizes()) {
    for (const frame of pickFrames()) {
      const key = `${frame}-${sizeName(size)}`;
      const n = (notes[key] = { url: null, steps: [], errors: [] });
      const note = (k, v) => n.steps.push(`${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
      const sc = SCENARIOS[frame];
      const ctx = await browser.newContext(contextFor(size, { reducedMotion: "reduce" }));
      await ctx.addInitScript((kv) => {
        if (sessionStorage.getItem("__v14_seeded")) return;
        sessionStorage.setItem("__v14_seeded", "1");
        for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
      }, store(sc.home));
      const page = await ctx.newPage();
      page.on("pageerror", (e) => n.errors.push(e.message.slice(0, 200)));
      page.on("console", (m) => m.type() === "error" && n.errors.push(`console: ${m.text().slice(0, 200)}`));
      const step = async (what, fn) => {
        try {
          await fn();
          n.steps.push(`ok ${what}`);
        } catch (e) {
          n.steps.push(`FAILED ${what}: ${String(e.message).split("\n")[0].slice(0, 200)}`);
        }
      };
      const publish = sc.prepare ? await sc.prepare(page) : null;
      n.url = `${BASE}${sc.url}`;
      await step("open", () => page.goto(n.url, { waitUntil: "load", timeout: 30_000 }));
      await sc.setup(page, step, note, size, publish);
      await page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur());
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(SETTLE);
      n.hscroll = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      const png = await page.screenshot();
      writeFileSync(join(OUT, "actual", `${key}.webp`), await tools.webp(png, 0.86));
      const refFile = join(OUT, "reference", `${key}.webp`);
      const ref = existsSync(refFile) ? await tools.png(readFileSync(refFile)) : null;
      writeFileSync(join(OUT, "compare", `${key}.webp`), await tools.composite(ref, png, { title: key }));
      const failed = n.steps.filter((s) => s.startsWith("FAILED"));
      rows.push({ key, frame, size: sizeName(size), url: sc.url, failed, errors: n.errors.length, hscroll: n.hscroll });
      console.log(`${key}  ${failed.length ? `${failed.length} step(s) failed` : "ok"}${n.errors.length ? `, ${n.errors.length} error(s)` : ""}${n.hscroll > 0 ? `, scrolls sideways ${n.hscroll}px` : ""}`);
      await ctx.close();
    }
  }
} finally {
  await tools.close();
  await browser.close();
  stop();
}

writeFileSync(join(OUT, "actual", "notes.json"), JSON.stringify(notes, null, 1));
const when = new Date().toISOString().slice(0, 16).replace("T", " ");
const md = [
  "# v14: the app beside the approved synthesis",
  "",
  `Generated ${when} UTC by \`scripts/v14-app-shots.mjs\` against ${FIXTURE ? "fixture data (clock 2026-10-13)" : `\`${DATA}\``}. Left: the prototype (\`reference/\`, from \`scripts/v14-reference-shots.mjs\`). Right: the app (\`actual/\`). Device scale 1, reduced motion.`,
  "",
  "Live values differ from the prototype's examples by design; judge composition, typography, colour, cartography and hierarchy. Steps and errors per state: `actual/notes.json`.",
  "",
  "| State | Size | App address | Set-up | Sideways scroll |",
  "|---|---|---|---|---|",
  ...rows.map((r) => `| [${r.frame}](compare/${r.key}.webp) | ${r.size} | \`${r.url}\` | ${r.failed.length ? `${r.failed.length} step(s) failed: ${r.failed.map((f) => f.replace(/\|/g, "/")).join("; ")}` : "ok"}${r.errors ? `; ${r.errors} console error(s)` : ""} | ${r.hscroll > 0 ? `${r.hscroll} px` : "none"} |`),
  "",
  ...pickFrames().flatMap((f) => [`## ${f}`, "", ...pickSizes().map((s) => `![${f} ${sizeName(s)}](compare/${f}-${sizeName(s)}.webp)`), ""]),
].join("\n");
writeFileSync(join(OUT, "compare.md"), md);
console.log(`wrote ${join(OUT, "compare.md")}`);
