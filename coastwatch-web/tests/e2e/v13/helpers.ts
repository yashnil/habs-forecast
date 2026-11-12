import AxeBuilder from "@axe-core/playwright";
import { expect, type BrowserContext, type Page, type Route } from "@playwright/test";

export const MOSS = "port-592";
export const BODEGA = "port-473";

export const portRecord = (o: { home?: string | null; last?: string | null; fisheries?: string[] } = {}) =>
  JSON.stringify({ v: 1, place: o.home ?? null, for: "all", species: "any", lastPort: o.last ?? o.home ?? null, fisheries: o.fisheries ?? ["dungeness_commercial"], vessel: null });

/** Put values in the device's storage before the first page of the context loads (and only then). */
export async function seed(ctx: BrowserContext, items: Record<string, string>) {
  await ctx.addInitScript((kv) => {
    if (localStorage.getItem("__e2e_seeded")) return;
    localStorage.setItem("__e2e_seeded", "1");
    for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
  }, items);
}

/** A returning reader: no opening, no tour. */
export const RETURNING = { "nx-opening-seen-v1": "1", "nx-tour-v1": "dismissed" };

export async function axe(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  return r.violations.map((v) => `${v.impact} ${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

export const noSidewaysScroll = async (page: Page) => expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);

/**
 * A second state of the published records for one port, as the server would send it after a refresh: the real
 * fixture answer with one record changed (`edit`), published later. Until `publish()` is called the real
 * answer (the first state) goes through untouched.
 */
export async function secondState(page: Page, port: string, edit: (inputs: Record<string, unknown>) => void) {
  let published = false;
  await page.route(`**/today/data?place=${port}`, async (route: Route) => {
    const res = await route.fetch();
    if (!published) return route.fulfill({ response: res });
    const j = (await res.json()) as { inputs: Record<string, unknown> };
    edit(j.inputs);
    j.inputs.dataGeneratedAt = "2026-10-13T05:30:00Z";
    await route.fulfill({ response: res, json: j });
  });
  return {
    publish: () => {
      published = true;
    },
  };
}

/** The commercial Dungeness status changes (the fixture says "delayed to at least Jan 1, 2027"). */
export const dungenessOpens = (inputs: Record<string, unknown>) => {
  const rec = (inputs.fisheries as { records: Array<{ fishery: string; headline: string; level: string }> }).records.find((x) => x.fishery === "dungeness_commercial")!;
  rec.headline = "Commercial season opens Jan 1, 2027 with a 30-fathom depth limit.";
  rec.level = "restricted";
};

/** Thirty-one minutes pass since this device's last check (the check is due again). */
export const halfAnHourLater = (page: Page) =>
  page.evaluate(() => {
    const k = "nx-alert-check-v1";
    const s = JSON.parse(localStorage.getItem(k) ?? "null");
    if (!s) return;
    s.at = new Date(Date.now() - 31 * 60_000).toISOString();
    localStorage.setItem(k, JSON.stringify(s));
  });

export const stored = (page: Page, key: string) => page.evaluate((k) => JSON.parse(localStorage.getItem(k) ?? "null"), key);
