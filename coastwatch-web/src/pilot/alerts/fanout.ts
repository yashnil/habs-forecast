import type { PortToday } from "@/pilot/contracts";
import { diffFacts, snapshotOf, type AlertItem, type AlertTrigger, type PlaceSnapshot, type SeenFact } from "./diff";

/**
 * alerts-fanout: after a data refresh, compare each active subscription's last-seen fact versions
 * with the newest PortToday facts, write inbox alerts, and (when VAPID is configured) send web push.
 *
 * Idempotent by construction: a delivery is first *claimed* in notifications_sent, whose primary key
 * is subscription × fact id × fact version. A rerun (cron retry, two overlapping runs) finds the claim
 * and sends nothing. The fact state is advanced after the claims, so a crash between the two only
 * means the next run re-claims (and is refused) — never a duplicate.
 */
export type Subscription = { id: string; user_id: string; place_id: string; triggers: AlertTrigger[]; channel: "in_app" | "web_push" | "sms" };
export type FactStateRow = { subscription_id: string; fact_id: string; fact_version: string; fact: SeenFact["sig"]; seen_at: string };
export type PushTarget = { endpoint: string; p256dh: string; auth: string };

export interface FanoutStore {
  activeSubscriptions(): Promise<Subscription[]>;
  factState(subscriptionIds: string[]): Promise<FactStateRow[]>;
  /** insert into notifications_sent; false when that key was already claimed (nothing to send) */
  claim(sub: Subscription, a: AlertItem): Promise<boolean>;
  insertAlert(sub: Subscription, a: AlertItem): Promise<void>;
  saveFactState(rows: (FactStateRow & { user_id: string })[]): Promise<void>;
  pushTargets(userId: string): Promise<PushTarget[]>;
  forgetPushTarget(endpoint: string): Promise<void>;
}

export type PushSender = (target: PushTarget, payload: { title: string; body: string; url: string; tag: string }) => Promise<"sent" | "gone" | "failed">;

export type FanoutReport = {
  subscriptions: number;
  places_without_facts: string[];
  baselined: number;
  alerts_written: number;
  duplicates_skipped: number;
  push: { sent: number; gone: number; failed: number; skipped_not_configured: number };
  sms_not_sent: number;
};

function toSnapshot(place: string, rows: FactStateRow[]): PlaceSnapshot | null {
  if (!rows.length) return null;
  const facts: Record<string, SeenFact> = {};
  let seen = rows[0].seen_at;
  for (const r of rows) {
    facts[r.fact_id] = { v: r.fact_version, sig: r.fact };
    if (r.seen_at < seen) seen = r.seen_at;
  }
  return { place_id: place, seen_at: seen, facts };
}

export async function runFanout(store: FanoutStore, todays: PortToday[], push: PushSender | null, siteUrl: string, now: Date = new Date()): Promise<FanoutReport> {
  const iso = now.toISOString();
  const subs = await store.activeSubscriptions();
  const state = await store.factState(subs.map((s) => s.id));
  const byPlace = new Map(todays.map((t) => [t.place_id, t]));
  const report: FanoutReport = { subscriptions: subs.length, places_without_facts: [], baselined: 0, alerts_written: 0, duplicates_skipped: 0, push: { sent: 0, gone: 0, failed: 0, skipped_not_configured: 0 }, sms_not_sent: 0 };
  const missing = new Set<string>();

  for (const sub of subs) {
    const today = byPlace.get(sub.place_id);
    if (!today) {
      missing.add(sub.place_id);
      continue;
    }
    const prev = toSnapshot(sub.place_id, state.filter((r) => r.subscription_id === sub.id));
    if (!prev) report.baselined++;
    const items = diffFacts(prev, today, sub.triggers, iso);
    for (const a of items) {
      if (!(await store.claim(sub, a))) {
        report.duplicates_skipped++;
        continue;
      }
      await store.insertAlert(sub, a);
      report.alerts_written++;
      if (sub.channel === "web_push") {
        if (!push) {
          report.push.skipped_not_configured++;
          continue;
        }
        for (const target of await store.pushTargets(sub.user_id)) {
          const r = await push(target, { title: a.title, body: a.body.slice(0, 240), url: `${siteUrl.replace(/\/$/, "")}/account#inbox`, tag: a.fact_id });
          report.push[r]++;
          if (r === "gone") await store.forgetPushTarget(target.endpoint);
        }
      } else if (sub.channel === "sms") {
        // SMS delivery belongs to the Ask/SMS track (carrier registration pending); the inbox row is the record
        report.sms_not_sent++;
      }
    }
    const snap = snapshotOf(today, iso);
    await store.saveFactState(Object.entries(snap.facts).map(([fact_id, f]) => ({ subscription_id: sub.id, user_id: sub.user_id, fact_id, fact_version: f.v, fact: f.sig, seen_at: iso })));
  }
  report.places_without_facts = [...missing];
  return report;
}

/** In-memory store: tests, and a dry run of the fan-out without a database. */
export class MemoryFanoutStore implements FanoutStore {
  subs: Subscription[] = [];
  state: (FactStateRow & { user_id: string })[] = [];
  sent = new Set<string>();
  alerts: { user_id: string; subscription_id: string; item: AlertItem }[] = [];
  targets = new Map<string, PushTarget[]>();
  async activeSubscriptions() {
    return this.subs;
  }
  async factState(ids: string[]) {
    return this.state.filter((r) => ids.includes(r.subscription_id));
  }
  async claim(sub: Subscription, a: AlertItem) {
    const k = `${sub.id}|${a.fact_id}|${a.fact_version}`;
    if (this.sent.has(k)) return false;
    this.sent.add(k);
    return true;
  }
  async insertAlert(sub: Subscription, item: AlertItem) {
    this.alerts.push({ user_id: sub.user_id, subscription_id: sub.id, item });
  }
  async saveFactState(rows: (FactStateRow & { user_id: string })[]) {
    for (const r of rows) {
      this.state = this.state.filter((s) => !(s.subscription_id === r.subscription_id && s.fact_id === r.fact_id));
      this.state.push(r);
    }
  }
  async pushTargets(userId: string) {
    return this.targets.get(userId) ?? [];
  }
  async forgetPushTarget(endpoint: string) {
    for (const [u, list] of this.targets) this.targets.set(u, list.filter((t) => t.endpoint !== endpoint));
  }
}
