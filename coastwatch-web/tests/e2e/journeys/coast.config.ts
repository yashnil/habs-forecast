import { defineConfig } from "@playwright/test";

/**
 * The My Coast journey (tests/e2e/journeys/coast.spec.ts) on one production server with the main fixture and
 * recorded NOAA responses: `npm run data:fixture && npm run build && npx playwright test -c tests/e2e/journeys/coast.config.ts`.
 * CW_E2E_PORT_BASE moves it; NX_MEDIA=1 also writes review media to docs/coastwatch/v12/implementation/coast/.
 */
const PORT = Number(process.env.CW_E2E_PORT_BASE ?? 3200);

export default defineConfig({
  testDir: ".",
  testMatch: /coast\.spec\.ts$/,
  timeout: 60_000,
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
    cwd: "../../..",
    port: PORT,
    reuseExistingServer: false,
    timeout: 60_000,
    env: { CW_DATA_BASE_URL: "/data/fixture/v1", CW_CONDITIONS_FIXTURE: "tests/conditions-fixtures.json", CW_NWS_API: "1", CW_FIXED_NOW: "2026-10-13T06:00:00Z" },
  },
});
