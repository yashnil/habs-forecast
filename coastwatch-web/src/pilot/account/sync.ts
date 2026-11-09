"use client";

import type { SupabaseClient } from "@supabase/supabase-js";
import type { FisheryId } from "@/pilot/contracts";
import type { AlertTrigger } from "@/pilot/alerts/diff";
import type { InboxItem } from "@/pilot/alerts/inbox";
import type { Channel, LocalAccount, Role, VesselClass } from "./local";
import type { FeedbackRecord, TripLog } from "./records";
import { AGE_TEXT, LEGAL_VERSION } from "./consent";

/**
 * Signed in (accounts live/demo): the device copy is written to the person's own rows. Every call goes
 * through row-level security as that person (anon key + session), so it can only ever touch their
 * rows. Errors are returned, never thrown, so the page can say "kept on this device; not synced".
 */
export type SyncResult = { ok: true } | { ok: false; error: string };

const fail = (e: unknown): SyncResult => ({ ok: false, error: e instanceof Error ? e.message : String((e as { message?: string })?.message ?? e) });

export async function pushAccount(db: SupabaseClient, a: LocalAccount): Promise<SyncResult> {
  try {
    const { data: u } = await db.auth.getUser();
    const uid = u.user?.id;
    if (!uid) return { ok: false, error: "not signed in" };
    if (!a.age_13_plus) return { ok: false, error: "13+ attestation needed" };
    let r = await db.from("profiles").upsert({
      user_id: uid,
      role: a.role ?? "commercial_fisher",
      fisheries: a.fisheries,
      home_place_id: a.home_place_id,
      vessel_length_class: a.vessel_length_class,
      age_13_plus: true,
      age_attested_at: a.age_13_plus.at,
    });
    if (r.error) return fail(r.error);

    // saved places: the device list is the truth
    r = await db.from("saved_places").delete().eq("user_id", uid).not("place_id", "in", `(${a.saved_places.map((p) => `"${p}"`).join(",") || '""'})`);
    if (r.error) return fail(r.error);
    if (a.saved_places.length) {
      r = await db.from("saved_places").upsert(a.saved_places.map((place_id) => ({ user_id: uid, place_id })), { onConflict: "user_id,place_id", ignoreDuplicates: true });
      if (r.error) return fail(r.error);
    }

    // subscriptions: one row per place × channel; channels no longer chosen are deactivated
    const rows = a.alerts.flatMap((p) =>
      p.channels
        .filter((c) => c !== "sms" || a.sms)
        .map((channel) => ({
          user_id: uid,
          place_id: p.place_id,
          channel,
          triggers: p.triggers,
          active: p.triggers.length > 0 && a.saved_places.includes(p.place_id),
          ...(channel === "sms" && a.sms ? { sms_phone_e164: a.sms.phone_e164, sms_consent_at: a.sms.consent.at, sms_consent_text: a.sms.consent.text } : {}),
        })),
    ).filter((row) => row.triggers.length > 0);
    r = await db.from("alert_subscriptions").update({ active: false }).eq("user_id", uid);
    if (r.error) return fail(r.error);
    if (rows.length) {
      r = await db.from("alert_subscriptions").upsert(rows, { onConflict: "user_id,place_id,channel" });
      if (r.error) return fail(r.error);
    }
    return { ok: true };
  } catch (e) {
    return fail(e);
  }
}

/** On sign-in on a new device: the account's rows, as a device account (null when there is no profile yet). */
export async function pullAccount(db: SupabaseClient): Promise<Partial<LocalAccount> | null> {
  try {
    const [p, s, subs] = await Promise.all([
      db.from("profiles").select("role, fisheries, home_place_id, vessel_length_class, age_attested_at").maybeSingle(),
      db.from("saved_places").select("place_id").order("created_at"),
      db.from("alert_subscriptions").select("place_id, channel, triggers, active, sms_phone_e164, sms_consent_at, sms_consent_text"),
    ]);
    if (p.error || !p.data) return null;
    const alerts = new Map<string, { place_id: string; triggers: AlertTrigger[]; channels: Channel[] }>();
    let sms: LocalAccount["sms"] = null;
    for (const row of subs.data ?? []) {
      if (!row.active) continue;
      const cur = alerts.get(row.place_id) ?? { place_id: row.place_id, triggers: [] as AlertTrigger[], channels: [] as Channel[] };
      cur.triggers = [...new Set([...cur.triggers, ...(row.triggers as AlertTrigger[])])];
      cur.channels.push(row.channel as Channel);
      alerts.set(row.place_id, cur);
      if (row.channel === "sms" && row.sms_phone_e164 && row.sms_consent_at && row.sms_consent_text) sms = { phone_e164: row.sms_phone_e164, consent: { at: row.sms_consent_at, text: row.sms_consent_text } };
    }
    return {
      role: p.data.role as Role,
      fisheries: p.data.fisheries as FisheryId[],
      home_place_id: p.data.home_place_id,
      vessel_length_class: p.data.vessel_length_class as VesselClass | null,
      age_13_plus: { at: p.data.age_attested_at, text: AGE_TEXT },
      saved_places: (s.data ?? []).map((r) => r.place_id),
      alerts: [...alerts.values()],
      sms,
    };
  } catch {
    return null;
  }
}

export async function recordConsent(db: SupabaseClient, kind: "terms" | "privacy" | "age_13_plus" | "sms" | "web_push", granted: boolean, text: string): Promise<SyncResult> {
  try {
    const { data: u } = await db.auth.getUser();
    if (!u.user) return { ok: false, error: "not signed in" };
    const r = await db.from("consent_log").insert({ user_id: u.user.id, kind, granted, text_shown: text, document_version: LEGAL_VERSION });
    return r.error ? fail(r.error) : { ok: true };
  } catch (e) {
    return fail(e);
  }
}

export async function fetchInbox(db: SupabaseClient): Promise<InboxItem[]> {
  try {
    const r = await db.from("alerts").select("id, place_id, fact_id, fact_version, trigger, severity, title, body, basis, verified, source_name, source_url, as_of, created_at, read_at").order("created_at", { ascending: false }).limit(50);
    if (r.error || !r.data) return [];
    return r.data.map((a) => ({
      key: `${a.place_id}|${a.fact_id}|${a.fact_version}`,
      account_id: a.id,
      place_id: a.place_id,
      place_label: a.place_id,
      fact_id: a.fact_id,
      fact_version: a.fact_version,
      trigger: a.trigger,
      severity: a.severity,
      title: a.title,
      body: a.body,
      basis: a.basis,
      verified: a.verified,
      source: a.source_name ? { name: a.source_name, url: a.source_url } : null,
      as_of: a.as_of,
      created_at: a.created_at,
      read_at: a.read_at,
      origin: "account" as const,
    }));
  } catch {
    return [];
  }
}

export async function markAccountRead(db: SupabaseClient, ids: string[]): Promise<void> {
  if (!ids.length) return;
  try {
    await db.from("alerts").update({ read_at: new Date().toISOString() }).in("id", ids).is("read_at", null);
  } catch {
    // the device copy is marked anyway
  }
}

export async function sendFeedback(db: SupabaseClient, f: FeedbackRecord): Promise<SyncResult> {
  try {
    const { data: u } = await db.auth.getUser();
    if (!u.user) return { ok: false, error: "not signed in" };
    const r = await db.from("feedback").insert({ user_id: u.user.id, fact_id: f.fact_id, place_id: f.place_id, useful: f.useful, note: f.note });
    return r.error ? fail(r.error) : { ok: true };
  } catch (e) {
    return fail(e);
  }
}

export async function sendTrip(db: SupabaseClient, t: TripLog): Promise<SyncResult> {
  try {
    const { data: u } = await db.auth.getUser();
    if (!u.user) return { ok: false, error: "not signed in" };
    const r = await db.from("trip_logs").insert({ user_id: u.user.id, trip_date: t.trip_date, place_id: t.place_id, fishery: t.fishery, outcome_band: t.outcome_band, observed_conditions: t.observed_conditions, recommendation_useful: t.recommendation_useful });
    return r.error ? fail(r.error) : { ok: true };
  } catch (e) {
    return fail(e);
  }
}

export async function savePushEndpoint(db: SupabaseClient, sub: PushSubscriptionJSON): Promise<SyncResult> {
  try {
    const { data: u } = await db.auth.getUser();
    if (!u.user) return { ok: false, error: "not signed in" };
    if (!sub.endpoint || !sub.keys?.p256dh || !sub.keys?.auth) return { ok: false, error: "incomplete subscription" };
    const r = await db.from("push_subscriptions").upsert({ user_id: u.user.id, endpoint: sub.endpoint, p256dh: sub.keys.p256dh, auth: sub.keys.auth, user_agent: navigator.userAgent.slice(0, 300) }, { onConflict: "endpoint" });
    return r.error ? fail(r.error) : { ok: true };
  } catch (e) {
    return fail(e);
  }
}

/** Deletes the auth user server-side (delete_my_account cascades to every row), then signs out. */
export async function deleteAccount(db: SupabaseClient): Promise<SyncResult> {
  try {
    const r = await db.rpc("delete_my_account");
    if (r.error) return fail(r.error);
    await db.auth.signOut();
    return { ok: true };
  } catch (e) {
    return fail(e);
  }
}
