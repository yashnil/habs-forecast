import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * Old links land on the same view in v12 (ported from explore.spec "v2 links still open their view",
 * navigation.spec "v2 links open the same view", m6 "the view is in the link", pilot-front-door "existing deep
 * links still open Explore"). nx-shell.spec checks the redirect responses themselves; these check what opens.
 */
test.use({ storageState: { cookies: [], origins: [] } });

test("a v2 Explore link naming the forecast opens Explore on C-HARM, with its words", async ({ page }) => {
  await page.goto(`${OK}/?region=monterey_bay&port=550&layer=forecast`);
  await expect(page).toHaveURL(/\/explore\?/);
  // v15.1: the page heading names the layer on show (Explore California Waters: C-HARM)
  await expect(page.getByTestId("nx-margin").getByRole("heading", { level: 1, name: "Explore California Waters: C-HARM" })).toBeVisible();
  await expect(page.getByTestId("nx-margin")).toContainText("not toxin in seafood");
});

test("a v2 station link (/bloom with a year) opens that station's record on that year", async ({ page }) => {
  await page.goto(`${OK}/bloom?station=HABs-SantaCruzWharf&var=pDA&year=2025`);
  await expect(page).toHaveURL(/\/research\?/);
  await expect(page.getByTestId("research-title")).toHaveText("Santa Cruz Wharf");
  await expect(page.getByTestId("research-record")).toContainText("2025");
});

test("a research-mode My Coast link opens Research near that place", async ({ page }) => {
  await page.goto(`${OK}/coast?mode=research&place=port-592`);
  await expect(page).toHaveURL(/\/research\?place=port-592/);
  await expect(page.getByTestId("nx-research")).toBeVisible();
  // v15.1: Research opens on the most useful station near the place and says so ("… stations near Moss Landing")
  await expect(page.getByTestId("nx-margin")).toContainText(/stations? near Moss Landing/);
});

test("a campaign tag on / keeps the front door; a reload of Explore stays on Explore", async ({ page }) => {
  await page.goto(`${OK}/?utm_source=newsletter`);
  await expect(page.getByTestId("nx-home")).toBeVisible();
  await page.goto(`${OK}/explore?layer=charm`);
  await page.reload();
  await expect(page).toHaveURL(/\/explore\?layer=charm/);
  await expect(page.getByTestId("nx-margin").getByRole("heading", { level: 1, name: "Explore California Waters: C-HARM" })).toBeVisible();
});

test("About and Sources are one tap from the menu on every page (navigation.spec)", async ({ page }) => {
  for (const url of ["/explore", "/coast", "/research", "/today?place=port-592"]) {
    await page.goto(`${OK}${url}`);
    await page.getByTestId("nx-menu-button").click();
    await expect(page.getByRole("link", { name: "About" }), url).toHaveAttribute("href", "/about");
    await expect(page.getByRole("link", { name: "Sources" }), url).toHaveAttribute("href", "/sources");
  }
  await page.getByRole("link", { name: "About" }).click();
  await expect(page).toHaveURL(`${OK}/about`);
  await expect(page.getByTestId("about-independence")).toContainText("not affiliated with or endorsed by NOAA, CDFW, or CDPH");
});
