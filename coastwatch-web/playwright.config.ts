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
  env: { CW_DATA_BASE_URL: base },
});

const BASE = Number(process.env.CW_E2E_PORT_BASE ?? 3200); // tests/e2e/ports.ts

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 45_000,
  fullyParallel: true,
  reporter: [["list"]],
  use: {
    viewport: { width: 1440, height: 900 },
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
