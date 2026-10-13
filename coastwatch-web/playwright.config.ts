import { defineConfig } from "@playwright/test";

/**
 * End-to-end tests run against a production build with deterministic fixture data:
 *   npm run test:e2e   (copies fixtures, builds, then runs)
 * Three servers share one build and differ only in which data they read.
 */
const start = (port: number, base: string) => ({
  command: `npx next start -p ${port}`,
  port,
  reuseExistingServer: false,
  timeout: 60_000,
  // ocean conditions are replayed from recorded NOAA responses (2026-10-13), never fetched live
  env: { CW_DATA_BASE_URL: base, CW_CONDITIONS_FIXTURE: "tests/conditions-fixtures.json", CW_FIXED_NOW: "2026-10-13T06:00:00Z" },
});

const BASE = Number(process.env.CW_E2E_PORT_BASE ?? 3200); // tests/e2e/ports.ts

// Returning-visitor state for every test server: the first-visit introduction on the map is
// already dismissed, so it never covers what a test clicks, and the presentation is Research (the
// detailed layout the earlier suites were written for). Coastal mode is tested explicitly
// (tests/e2e/coastal.spec.ts); m8, welcome and tour open fresh contexts.
const dismissedIntro = {
  cookies: [],
  origins: [0, 1, 2, 3].map((i) => ({
    origin: `http://localhost:${BASE + i}`,
    localStorage: [
      { name: "cw-intro-dismissed-v1", value: "1" },
      { name: "cw-mode-v1", value: "research" },
    ],
  })),
};

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 45_000,
  fullyParallel: true,
  // a failure keeps its evidence: the HTML report, a trace (DOM, network, console) and a screenshot
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  use: {
    storageState: dismissedIntro,
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    // Local runs use installed Chrome (GPU WebGL renders the vector basemap); CI uses bundled Chromium.
    channel: process.env.CI ? undefined : "chrome",
  },
  webServer: [
    start(BASE + 0, "/data/fixture/v1"),
    start(BASE + 1, "/data/fixture-failed/v1"),
    start(BASE + 2, "/data/does-not-exist/v1"),
    start(BASE + 3, "/data/compat-m2/v1"),
  ],
});
