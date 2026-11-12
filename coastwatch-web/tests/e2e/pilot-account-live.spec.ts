import { expect, test } from "@playwright/test";

/**
 * Opt-in: the account flow against a real Supabase (the local stack, `npx supabase@2.121.0 start`
 * from the repo root, ports 563xx). Skipped unless PILOT_LIVE_BASE points at a Neritic server started
 * with that stack's env (docs/coastwatch/pilot/accounts.md, "Run the local stack"):
 *   email one-time code (read from the stack's Mailpit) → profile/ports/subscriptions in the database
 *   → alerts-fanout writes an account alert → inbox shows it → account deletion removes everything.
 */
const BASE = process.env.PILOT_LIVE_BASE;
const API = process.env.PILOT_SUPABASE_URL ?? "http://127.0.0.1:56321";
const MAIL = process.env.PILOT_MAILPIT_URL ?? "http://127.0.0.1:56324";
const SERVICE = process.env.SUPABASE_SERVICE_ROLE_KEY ?? "";
const SECRET = process.env.ALERTS_FANOUT_SECRET ?? "";

test.skip(!BASE || !SERVICE || !SECRET, "set PILOT_LIVE_BASE, SUPABASE_SERVICE_ROLE_KEY and ALERTS_FANOUT_SECRET to run against a local Supabase");

const rest = (path: string, init: RequestInit = {}) =>
  fetch(`${API}/rest/v1/${path}`, { ...init, headers: { apikey: SERVICE, authorization: `Bearer ${SERVICE}`, "content-type": "application/json", prefer: "return=representation", ...(init.headers ?? {}) } }).then((r) => r.json());

async function codeFor(email: string): Promise<string> {
  for (let i = 0; i < 40; i++) {
    const list = (await fetch(`${MAIL}/api/v1/search?query=${encodeURIComponent(`to:${email}`)}`).then((r) => r.json())) as { messages: { ID: string }[] };
    if (list.messages?.length) {
      const m = (await fetch(`${MAIL}/api/v1/message/${list.messages[0].ID}`).then((r) => r.json())) as { Text: string; HTML: string };
      const code = `${m.Text} ${m.HTML}`.match(/\b(\d{6})\b/);
      if (code) return code[1];
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("no sign-in email");
}

test("sign in with an email code, sync, receive a server alert, delete the account", async ({ page, request }) => {
  test.setTimeout(90_000);
  const email = `pilot-${Date.now()}@example.test`;
  await page.goto(`${BASE}/account/pilot`);
  await expect(page.getByTestId("account-page")).toHaveAttribute("data-mode", "signed_out");
  await expect(page.getByTestId("readiness-accounts")).toHaveText("Pilot demo");

  await page.getByTestId("signin-email").fill(email);
  await page.getByTestId("consent-age").check();
  await page.getByTestId("signin-send").click();
  await expect(page.getByTestId("signin-msg")).toContainText("We sent a 6-digit code");
  await page.getByTestId("signin-code").fill(await codeFor(email));
  await page.getByTestId("signin-verify").click();
  await expect(page.getByTestId("signin-msg")).toContainText("Signed in. Your ports and alerts are saved to your account.");
  await expect(page.getByTestId("account-page")).toHaveAttribute("data-mode", "account");

  await page.getByTestId("save-port-592").click();
  await expect(page.getByTestId("sync-note")).toHaveText("Saved to your account.");
  const [user] = (await fetch(`${API}/auth/v1/admin/users?per_page=1000`, { headers: { apikey: SERVICE, authorization: `Bearer ${SERVICE}` } })
    .then((r) => r.json())
    .then((r: { users: { id: string; email: string }[] }) => r.users.filter((u) => u.email === email))) as { id: string }[];
  const uid = user.id;
  expect(await rest(`profiles?select=home_place_id,age_13_plus&user_id=eq.${uid}`)).toEqual([{ home_place_id: "port-592", age_13_plus: true }]);
  const subs = (await rest(`alert_subscriptions?select=id,channel,triggers,active&user_id=eq.${uid}`)) as { id: string; channel: string; active: boolean }[];
  expect(subs).toEqual([expect.objectContaining({ channel: "in_app", active: true })]);
  const consents = (await rest(`consent_log?select=kind&user_id=eq.${uid}`)) as { kind: string }[];
  expect(consents.map((c) => c.kind)).toContain("age_13_plus");

  // first fan-out keeps a baseline; then the stored version is made older, as if the status changed since
  const fan = () => request.post(`${BASE}/api/alerts/fanout`, { headers: { authorization: `Bearer ${SECRET}` } }).then((r) => r.json());
  expect(await request.post(`${BASE}/api/alerts/fanout`, { headers: { authorization: "Bearer wrong" } }).then((r) => r.status())).toBe(401);
  const first = await fan();
  expect(first).toMatchObject({ readiness: "demo", facts: "sample", alerts_written: 0 });
  await rest(`alert_fact_state?subscription_id=eq.${subs[0].id}&fact_id=eq.port-592:status:dungeness_commercial`, {
    method: "PATCH",
    body: JSON.stringify({ fact_version: "00000000", fact: { kind: "status", severity: "red", headline: "Commercial Dungeness: delayed to at least Dec 15", value: null, as_of: null } }),
  });
  const second = await fan();
  expect(second.alerts_written).toBe(1);
  expect((await fan()).alerts_written).toBe(0); // idempotent

  await page.reload();
  const item = page.locator('[data-testid="inbox-item"]').filter({ hasText: "from your account" });
  await expect(item).toHaveCount(1);
  await expect(item).toContainText("Commercial Dungeness");

  await page.getByTestId("delete-account").click();
  await page.getByTestId("confirm-delete-yes").click();
  await expect(page.getByTestId("data-msg")).toHaveText("Your account and this device's copy are deleted.");
  expect(await rest(`alerts?select=id&user_id=eq.${uid}`)).toEqual([]);
  expect(await rest(`saved_places?select=id&user_id=eq.${uid}`)).toEqual([]);
});
