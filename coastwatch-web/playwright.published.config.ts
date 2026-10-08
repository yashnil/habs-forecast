import { defineConfig } from "@playwright/test";

/**
 * Production-equivalent check: a production build reading the *published* dataset.
 *   PUBLISHED_DATA_URL=https://raw.githubusercontent.com/<owner>/<repo>/coastwatch-data/v1 \
 *     npm run build && npx playwright test -c playwright.published.config.ts
 */
const url = process.env.PUBLISHED_DATA_URL ?? "https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data/v1";

export default defineConfig({
  testDir: "tests/published",
  timeout: 60_000,
  // The app's server caches the manifest (stale-while-revalidate, 5 min) and GitHub Pages
  // caches for 10 min, so the first request after a new publish can render the previous
  // (still complete, correctly dated) version while the test compares with the new one.
  retries: 1,
  reporter: [["list"]],
  use: { viewport: { width: 1440, height: 900 }, channel: process.env.CI ? undefined : "chrome" },
  webServer: { command: "npx next start -p 3300", port: 3300, reuseExistingServer: false, env: { CW_DATA_BASE_URL: url } },
});
