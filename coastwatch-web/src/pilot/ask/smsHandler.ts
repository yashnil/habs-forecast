import { portById } from "./places";
import { homeCommand, keywordOf, SMS_TEXT } from "./sms";
import type { AskAnswer, AskRequest } from "./types";

/**
 * One inbound text → one reply (or none). Pure apart from the injected store and answerer, so the
 * whole conversation logic is unit-tested without Twilio.
 *
 * Opt-out and home port live in an SmsStore. The default is in-memory (per instance, lost on
 * redeploy): fine for the trial with team phones, NOT for production. Production needs a durable
 * store (a Supabase table owned by the accounts track) — see docs/coastwatch/pilot/sms.md.
 */

export interface SmsStore {
  isOptedOut(phone: string): Promise<boolean>;
  setOptedOut(phone: string, out: boolean): Promise<void>;
  homePort(phone: string): Promise<string | null>;
  setHomePort(phone: string, placeId: string): Promise<void>;
}

export class MemorySmsStore implements SmsStore {
  private out = new Set<string>();
  private home = new Map<string, string>();
  async isOptedOut(p: string) {
    return this.out.has(p);
  }
  async setOptedOut(p: string, v: boolean) {
    if (v) this.out.add(p);
    else this.out.delete(p);
  }
  async homePort(p: string) {
    return this.home.get(p) ?? null;
  }
  async setHomePort(p: string, id: string) {
    this.home.set(p, id);
  }
}

export type SmsDeps = {
  store: SmsStore;
  ask: (req: AskRequest) => Promise<AskAnswer>;
  allow: (phone: string) => boolean;
  /** Twilio Advanced Opt-Out answers STOP/START/HELP itself; we then stay silent but still record. */
  twilioHandlesKeywords?: boolean;
};

export async function handleInboundSms(from: string, body: string, deps: SmsDeps): Promise<string | null> {
  const kw = keywordOf(body);
  if (kw === "stop") {
    await deps.store.setOptedOut(from, true);
    return deps.twilioHandlesKeywords ? null : SMS_TEXT.stop;
  }
  if (kw === "start") {
    await deps.store.setOptedOut(from, false);
    return deps.twilioHandlesKeywords ? null : SMS_TEXT.start;
  }
  if (kw === "help") return deps.twilioHandlesKeywords ? null : SMS_TEXT.help;
  if (await deps.store.isOptedOut(from)) return null; // never text an opted-out number
  if (!deps.allow(from)) return SMS_TEXT.rateLimited;

  const home = homeCommand(body);
  if (home === "invalid") return SMS_TEXT.needPlace;
  if (home) {
    await deps.store.setHomePort(from, home.place_id);
    return SMS_TEXT.homeSet(home.label);
  }
  const saved = await deps.store.homePort(from);
  const a = await deps.ask({ question: body, place: portById(saved)?.place_id ?? null, channel: "sms" });
  if (a.outcome === "need_place" && !a.safety) return SMS_TEXT.needPlace;
  return a.sms;
}
