import { timingSafeEqual } from "node:crypto";
import { NextResponse } from "next/server";
import { readiness } from "@/pilot/readiness";
import { runFanout } from "@/pilot/alerts/fanout";
import { pushSender, serviceClient, SupabaseFanoutStore } from "@/pilot/alerts/server";
import { loadPortTodays } from "@/pilot/alerts/source";

export const dynamic = "force-dynamic";

/**
 * POST /api/alerts/fanout  (Authorization: Bearer $ALERTS_FANOUT_SECRET)
 *
 * Called by pg_cron + pg_net after each data refresh (supabase/migrations/…_pilot_accounts.sql,
 * commented schedule). Computes PortToday facts for every subscribed place, diffs them against the
 * versions each subscription last saw, writes inbox alerts and sends web push when VAPID is set.
 * Idempotent: safe to call any number of times (notifications_sent is the key).
 *
 * Without Supabase service credentials it does nothing and says so (503, "not_configured").
 */
const noStore = { "Cache-Control": "no-store" };

function authorized(req: Request): boolean {
  const secret = process.env.ALERTS_FANOUT_SECRET;
  if (!secret) return false;
  const got = Buffer.from(req.headers.get("authorization") ?? "");
  const want = Buffer.from(`Bearer ${secret}`);
  return got.length === want.length && timingSafeEqual(got, want);
}

export async function POST(req: Request) {
  if (!process.env.ALERTS_FANOUT_SECRET) return NextResponse.json({ readiness: "not_configured", error: "ALERTS_FANOUT_SECRET is not set" }, { status: 503, headers: noStore });
  if (!authorized(req)) return NextResponse.json({ error: "unauthorized" }, { status: 401, headers: noStore });
  const db = serviceClient();
  if (!db || readiness("alerts") === "not_configured") return NextResponse.json({ readiness: "not_configured", error: "Supabase is not configured" }, { status: 503, headers: noStore });

  const { todays, stub } = await loadPortTodays();
  // sample facts must never reach a real person's phone
  const push = stub ? null : pushSender();
  const site = process.env.CW_SITE_URL ?? new URL(req.url).origin;
  try {
    const report = await runFanout(new SupabaseFanoutStore(db), todays, push, site);
    return NextResponse.json({ readiness: readiness("alerts"), web_push: push ? readiness("web_push") : "not_configured", facts: stub ? "sample" : "published", ...report }, { headers: noStore });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500, headers: noStore });
  }
}
