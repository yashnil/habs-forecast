import { defineConfig } from "@playwright/test";

/**
 * The v12 shell and chart suite (tests/e2e/nx-shell*.spec.ts) on one production server with the main
 * fixture: `npm run build && npx playwright test -c playwright.nx-shell.config.ts`. CW_E2E_PORT_BASE moves it.
 * Storage starts empty: these tests are about what a first and a returning visit see.
 */
const PORT = Number(process.env.CW_E2E_PORT_BASE ?? 3200);

export default defineConfig({
  testDir: "tests/e2e",
  testMatch: /nx-shell.*\.spec\.ts$/,
  timeout: 45_000,
  fullyParallel: true,
  workers: 2,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${PORT}`,
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    channel: process.env.CI ? undefined : "chrome",
  },
  webServer: {
    command: `npx next start -p ${PORT}`,
    port: PORT,
    reuseExistingServer: false,
    timeout: 60_000,
    env: { CW_DATA_BASE_URL: "/data/fixture/v1", CW_FIXED_NOW: "2026-10-13T06:00:00Z" },
  },
});
