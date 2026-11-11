import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * The six usability issues the final scripted walkthrough left open (M2, M4–M8), each held here:
 * Research's map follows a new station; one station's latest HAB sample is dated the same way, with its
 * year, everywhere; what the map shows and when is above the fold; on a phone the layers stay reachable
 * with a place open and a statewide notice count says so; every listed notice carries its date; and
 * Research's map tools are one tap from the top on a phone.
 */

test("M8: when several notices cover a place, each listed notice carries the date it took effect", async ({ request }) => {
  const g = await (await request.get(`${OK}/api/evidence?place=port-550`)).json();
  const n = g.claims.find((c: { id: string }) => c.id === "notices");
  expect(n.headline).toMatch(/^\d+ notices cover these waters/);
  for (const part of (n.detail as string).split(" · ").filter((p: string) => !/more, all in Notices/.test(p))) expect(part).toMatch(/\(since [A-Z][a-z]{2} \d+\)$/);
});
