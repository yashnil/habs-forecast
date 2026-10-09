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
    start(3200, "/data/fixture/v1"),
    start(3201, "/data/fixture-failed/v1"),
    start(3202, "/data/does-not-exist/v1"),
    start(3203, "/data/compat-m2/v1"),
  ],
});
