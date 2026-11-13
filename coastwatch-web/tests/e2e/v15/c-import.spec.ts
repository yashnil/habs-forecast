import { expect, test } from "@playwright/test";
import { createTripsStore } from "@/nx/trips/store";
import { OK } from "../ports";
import { NS, newContext, seedTrip, tripKeys } from "./helpers";

/**
 * Import, restore and rename from the history (DIRECTION §12, §16; the review's fix round): a file is read and checked
 * on the device before any consent; the consent states the real counts; a re-imported backup adds nothing; a trip with
 * an id already used by a different trip comes in as "… (imported)"; Rename goes straight into a focused field.
 */
test.use({ storageState: { cookies: [], origins: [] } });

/** The trips export the store writes for this storage (what the reader's Export downloads). */
function exportOf(storage: Record<string, string>): string {
  const map = new Map(Object.entries(storage));
  const kv = { getItem: (k: string) => map.get(k) ?? null, setItem: (k: string, v: string) => void map.set(k, v), removeItem: (k: string) => void map.delete(k), key: (i: number) => [...map.keys()][i] ?? null, get length() { return map.size; } };
  return createTripsStore(kv).exportAll();
}

const file = (name: string, text: string) => ({ name, mimeType: "application/json", buffer: Buffer.from(text) });

test("import: a broken or foreign file says so and never asks for consent", async ({ browser }) => {
  const seeded = await seedTrip({ id: "c-one" });
  const { ctx, page, errors } = await newContext(browser, undefined, seeded.storage);
  await page.goto(`${OK}/coast/trips`);
  await expect(page.getByTestId("history-row")).toHaveCount(1);
  for (const text of ["{not json", JSON.stringify({ app: "some-other-app", trips: [{ id: "x" }] })]) {
    await page.getByTestId("trips-import-file").setInputFiles(file("backup.json", text));
    await expect(page.getByTestId("trips-import-report")).toContainText("not a Neritic trips export");
    await expect(page.getByTestId("trips-import-confirm")).toHaveCount(0);
    await page.getByTestId("trips-import-close").click();
  }
  await expect(page.getByTestId("history-row")).toHaveCount(1);
  expect(errors).toEqual([]);
  await ctx.close();
});

test("import: re-importing a backup adds nothing, and says the trips are already on this device", async ({ browser }) => {
  const seeded = await seedTrip({ id: "c-one" });
  const { ctx, page } = await newContext(browser, undefined, seeded.storage);
  await page.goto(`${OK}/coast/trips`);
  await expect(page.getByTestId("history-row")).toHaveCount(1);
  const before = await tripKeys(page);
  await page.getByTestId("trips-import-file").setInputFiles(file("backup.json", exportOf(seeded.storage)));
  await expect(page.getByTestId("trips-import-report")).toContainText("1 trip is already on this device");
  await expect(page.getByTestId("trips-import-confirm")).toHaveCount(0);
  await page.getByTestId("trips-import-close").click();
  await expect(page.getByTestId("history-row")).toHaveCount(1);
  expect(await tripKeys(page)).toEqual(before);
  await ctx.close();
});

test("import: the consent states the real counts; a same-id different trip comes in as a copy named (imported)", async ({ browser }) => {
  const seeded = await seedTrip({ id: "c-one" });
  const fresh = await seedTrip({ id: "c-two", edit: (t) => ({ ...t, name: "Second string" }) });
  const changed = await seedTrip({ id: "c-one", edit: (t) => ({ ...t, notes: "edited on the other phone" }) });
  const backup = JSON.parse(exportOf(fresh.storage));
  backup.trips.push(...JSON.parse(exportOf(changed.storage)).trips, ...JSON.parse(exportOf(seeded.storage)).trips, { schema: 1, name: "Torn record" });
  backup.snapshots.push(...JSON.parse(exportOf(changed.storage)).snapshots);
  const { ctx, page } = await newContext(browser, undefined, seeded.storage);
  await page.goto(`${OK}/coast/trips`);
  await expect(page.getByTestId("history-row")).toHaveCount(1);
  await page.getByTestId("trips-import-file").setInputFiles(file("phone-backup.json", JSON.stringify(backup)));
  const counts = page.getByTestId("trips-import-counts");
  await expect(counts).toContainText("1 new trip to add");
  await expect(counts).toContainText("separate copy");
  await expect(counts).toContainText("1 trip is already on this device");
  await expect(counts).toContainText("1 record cannot be read");
  await page.getByTestId("trips-import-confirm").click();
  await expect(page.getByTestId("trips-import-report")).toContainText("1 trip came in as a separate copy");
  await page.getByTestId("trips-import-close").click();
  await expect(page.getByTestId("history-row")).toHaveCount(3);
  await expect(page.getByTestId("history-row").filter({ hasText: "Moss Landing, Dungeness crab (imported)" })).toHaveCount(1);
  await expect(page.getByTestId("history-row").filter({ hasText: "Second string" })).toHaveCount(1);
  // the original is untouched, and the copy carries its own planning snapshot
  const keys = await tripKeys(page);
  expect(keys).toEqual(expect.arrayContaining([`${NS}trip:c-one`, `${NS}snap:c-one-plan`, `${NS}trip:c-two`]));
  expect(keys.filter((k) => k.startsWith(`${NS}snap:`)).length).toBe(3);
  await ctx.close();
});

test("rename from the row menu goes straight into a focused field; delete offers Undo", async ({ browser }) => {
  const one = await seedTrip({ id: "c-one" });
  const two = await seedTrip({ id: "c-two", edit: (t) => ({ ...t, name: "Second string" }) });
  const index = JSON.stringify({ ...JSON.parse(one.storage[`${NS}index`]), ids: ["c-two", "c-one"] });
  const { ctx, page } = await newContext(browser, undefined, { ...one.storage, ...two.storage, [`${NS}index`]: index });
  await page.goto(`${OK}/coast/trips`);
  const row = page.getByTestId("history-row").filter({ hasText: "Second string" });
  await row.getByRole("button", { name: /More for/ }).click();
  await page.getByRole("button", { name: "Rename", exact: true }).click();
  const field = page.getByTestId("history-rename");
  await expect(field).toBeFocused();
  await page.keyboard.type("Opening day");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("history-row").filter({ hasText: "Opening day" })).toHaveCount(1);
  await page.getByTestId("history-row").filter({ hasText: "Opening day" }).getByRole("button", { name: /More for/ }).click();
  await page.getByTestId("history-delete").click();
  await expect(page.getByTestId("history-undo")).toContainText("Deleted “Opening day”");
  await page.getByTestId("history-undo").getByRole("button", { name: "Undo" }).click();
  await expect(page.getByTestId("history-row")).toHaveCount(2);
  await ctx.close();
});
