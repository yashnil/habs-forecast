import { readiness } from "@/pilot/readiness";
import { allow, serveAsk } from "@/pilot/ask/service";
import { twiml, validTwilioRequest } from "@/pilot/ask/sms";
import { handleInboundSms, MemorySmsStore } from "@/pilot/ask/smsHandler";

export const dynamic = "force-dynamic";

/**
 * POST /api/sms — Twilio inbound-message webhook (application/x-www-form-urlencoded).
 * Validates X-Twilio-Signature with the auth token (WebCrypto HMAC-SHA1, no Twilio SDK), handles
 * STOP/START/HELP, place keywords and HOME, and replies with TwiML (≤ 2 segments).
 * Not configured (no TWILIO_* env): 503, nothing is processed.
 */

// per-instance; see smsHandler.ts — production needs a durable store
const store = new MemorySmsStore();

const xml = (body: string, status = 200) => new Response(body, { status, headers: { "Content-Type": "text/xml; charset=utf-8", "Cache-Control": "no-store" } });

function publicUrl(req: Request): string {
  const configured = process.env.NERITIC_SMS_WEBHOOK_URL;
  if (configured) return configured;
  const u = new URL(req.url);
  const proto = req.headers.get("x-forwarded-proto") ?? u.protocol.replace(":", "");
  const host = req.headers.get("x-forwarded-host") ?? req.headers.get("host") ?? u.host;
  return `${proto}://${host}${u.pathname}${u.search}`;
}

export async function POST(req: Request) {
  if (readiness("sms") === "not_configured") return new Response("SMS is not configured", { status: 503 });
  const form = await req.formData();
  const params: Record<string, string> = {};
  for (const [k, v] of form.entries()) if (typeof v === "string") params[k] = v;
  const ok = await validTwilioRequest(process.env.TWILIO_AUTH_TOKEN!, publicUrl(req), params, req.headers.get("x-twilio-signature"));
  if (!ok) return new Response("invalid signature", { status: 403 });
  const from = params.From ?? "";
  const body = params.Body ?? "";
  if (!from) return xml(twiml(null));
  const reply = await handleInboundSms(from, body, {
    store,
    allow: (phone) => allow("sms", phone),
    twilioHandlesKeywords: process.env.NERITIC_SMS_TWILIO_OPTOUT === "1",
    ask: (r) => serveAsk(r, { userKey: from, channel: "sms" }),
  });
  return xml(twiml(reply));
}
