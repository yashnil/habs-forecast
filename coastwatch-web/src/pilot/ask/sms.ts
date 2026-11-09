import { portInText } from "./places";
import { UNVERIFIED_LINE } from "./unverified";
import type { AskAnswer } from "./types";

/**
 * SMS for Ask Neritic: Twilio webhook signature validation (WebCrypto, no Twilio SDK), CTIA
 * keywords, place keywords, and the reply format (GSM-7, at most 2 segments = 306 characters,
 * with a short source tag). Pure functions; the route in src/app/api/sms wires them up.
 */

export const SMS_MAX = 306; // 2 concatenated GSM-7 segments × 153
export const PROGRAM = "Neritic Alerts";

/* ------------------------------------------------------------------------- GSM-7 charset */
const GSM_BASIC =
  "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà";
const GSM_EXT = "^{}\\[~]|€";
const REPLACE: [RegExp, string][] = [
  [/[‘’‛′]/g, "'"],
  [/[“”″]/g, '"'],
  [/[–—−]/g, "-"],
  [/…/g, "..."],
  [/m³/g, "m3"],
  [/\s?°\s?C\b/g, "C"],
  [/°/g, " deg"],
  [/≥/g, ">="],
  [/≤/g, "<="],
  [/×/g, "x"],
  [/[   ]/g, " "],
  [/·/g, "-"],
];

export function toGsm(s: string): string {
  let out = s;
  for (const [re, rep] of REPLACE) out = out.replace(re, rep);
  return [...out].filter((c) => GSM_BASIC.includes(c) || GSM_EXT.includes(c)).join("");
}

export function gsmLength(s: string): number {
  return [...s].reduce((n, c) => n + (GSM_EXT.includes(c) ? 2 : 1), 0);
}

export function segments(s: string): number {
  const n = gsmLength(s);
  return n <= 160 ? 1 : Math.ceil(n / 153);
}

/** Fit `body` + `tag` into SMS_MAX, cutting at a sentence, then a word. */
export function fitSms(body: string, tag: string, max = SMS_MAX): string {
  const b = toGsm(body).replace(/\s+/g, " ").trim();
  const t = toGsm(tag);
  const room = max - gsmLength(t) - 1;
  if (gsmLength(b) <= room) return `${b} ${t}`.trim();
  // a sentence ends at its full stop, not at a decimal point ("about 1.6x the typical level" stays whole)
  const sentences = b.match(/(?:[^.!?]|[.](?=\d))+[.!?]+/g) ?? [b];
  let acc = "";
  for (const s of sentences) {
    if (gsmLength(acc + s) > room) break;
    acc += s;
  }
  if (!acc) {
    acc = b.slice(0, room - 3);
    acc = acc.slice(0, acc.lastIndexOf(" ")) + "...";
  }
  return `${acc.trim()} ${t}`.trim();
}

const shortSource = (name: string) => name.replace(/^NOAA\s+/i, "").split(/\s+/).slice(0, 2).join(" ").slice(0, 18);
const md = (asOf: string | null) => (asOf ? `${Number(asOf.slice(5, 7))}/${Number(asOf.slice(8, 10))}` : "");
/** a phone's lines are the answer's own sentences (the tag carries the source and date), with the lab-sample caveat shortened */
const compact = (s: string) => s.replace(/ This is a past lab sample, not a forecast\./g, " (past sample)");

export function smsFromAnswer(a: Pick<AskAnswer, "outcome" | "preface" | "lines" | "citations">, safetySms?: string | null): string {
  // the closing "not yet verified" line is carried by the tag ("unverified") instead
  const kept = a.lines.filter((l) => !UNVERIFIED_LINE.test(l.text));
  const ordered = [...kept.filter((l) => l.primary), ...kept.filter((l) => !l.primary)];
  // the tag names the source of the first line the phone will show
  const lead = ordered.find((l) => l.cites.length);
  const first = lead ? a.citations.find((c) => c.fact_id === lead.cites[0]) : undefined;
  const facts = ordered.map((l) => compact(l.text));
  let body: string;
  let used: number;
  if (safetySms) [body, used] = [[safetySms, facts[0] ? `Info only: ${facts[0]}` : ""].join(" "), 1];
  else if (a.outcome === "answered" || a.outcome === "no_data") [body, used] = [facts.slice(0, 2).join(" "), 2];
  else [body, used] = [a.preface.join(" "), 0];
  const tagFor = (unverified: boolean) => (first ? `[Src: ${shortSource(first.source_name)} ${md(first.as_of)}${unverified ? ", unverified" : ""}]` : "");
  // fit with the longest tag, then say "unverified" only if a line that made it in rests on an unverified official record
  const draft = fitSms(body || a.preface.join(" "), tagFor(true));
  const shown = ordered.slice(0, used).filter((_, i) => draft.includes(toGsm(facts[i]).replace(/\s+/g, " ").trim().slice(0, 40)));
  const unverified = shown.some((l) => l.cites.some((id) => a.citations.find((c) => c.fact_id === id && c.basis === "official" && c.verified === false)));
  return fitSms(body || a.preface.join(" "), tagFor(unverified));
}

/* ------------------------------------------------------------------------------ keywords */
const STOP = new Set(["STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT", "OPTOUT", "REVOKE"]);
const START = new Set(["START", "UNSTOP", "YES", "SUBSCRIBE"]);
const HELP = new Set(["HELP", "INFO"]);

export type Keyword = "stop" | "start" | "help" | null;
export function keywordOf(body: string): Keyword {
  const w = body.trim().toUpperCase().replace(/[^A-Z]/g, "");
  if (STOP.has(w)) return "stop";
  if (START.has(w)) return "start";
  if (HELP.has(w)) return "help";
  return null;
}

export const SMS_TEXT = {
  help: `${PROGRAM}: answers about Monterey Bay & Half Moon Bay fisheries from public data. Not an official source; CDFW/CDPH decide. Text e.g. "MOSS crab status". Msg&data rates may apply. Reply STOP to cancel.`,
  stop: `${PROGRAM}: you're unsubscribed and won't get more messages. Reply START to resubscribe.`,
  start: `${PROGRAM}: you're subscribed. Ask e.g. "MOSS crab status" or set a home port: "HOME MOSS". Not an official source. Msg&data rates may apply. Reply HELP for help, STOP to cancel.`,
  needPlace: `${PROGRAM}: which port? Start with MOSS, MONTEREY, SANTACRUZ or HMB, e.g. "MOSS tide tomorrow". Or set one: "HOME MOSS".`,
  homeSet: (label: string) => `${PROGRAM}: home port set to ${label}. Ask e.g. "crab status" or "wind tomorrow".`,
  rateLimited: `${PROGRAM}: that's a lot of questions; please try again in an hour.`,
} as const;

/** "HOME MOSS" → the port; null if not a home command. */
export function homeCommand(body: string): { place_id: string; label: string } | "invalid" | null {
  const m = body.trim().match(/^home\s+(.+)$/i);
  if (!m) return null;
  const p = portInText(m[1]);
  return p ? { place_id: p.place_id, label: p.label } : "invalid";
}

/* -------------------------------------------------------------- Twilio request validation */
/**
 * X-Twilio-Signature = base64(HMAC-SHA1(authToken, url + Σ sorted(key + value))) for form posts.
 * `url` must be the exact public URL Twilio called (set NERITIC_SMS_WEBHOOK_URL behind proxies).
 */
export async function twilioSignature(authToken: string, url: string, params: Record<string, string>): Promise<string> {
  const data = url + Object.keys(params).sort().map((k) => k + params[k]).join("");
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey("raw", enc.encode(authToken), { name: "HMAC", hash: "SHA-1" }, false, ["sign"]);
  const sig = new Uint8Array(await crypto.subtle.sign("HMAC", key, enc.encode(data)));
  let bin = "";
  for (const b of sig) bin += String.fromCharCode(b);
  return btoa(bin);
}

export async function validTwilioRequest(authToken: string, url: string, params: Record<string, string>, header: string | null): Promise<boolean> {
  if (!header) return false;
  const expected = await twilioSignature(authToken, url, params);
  if (expected.length !== header.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) diff |= expected.charCodeAt(i) ^ header.charCodeAt(i);
  return diff === 0;
}

export function twiml(message: string | null): string {
  if (!message) return '<?xml version="1.0" encoding="UTF-8"?><Response></Response>';
  const esc = message.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  return `<?xml version="1.0" encoding="UTF-8"?><Response><Message>${esc}</Message></Response>`;
}

/** Remove a leading port keyword ("MOSS crab status" → "crab status"). */
export function stripPortKeyword(body: string): string {
  return body.replace(/^\s*(moss(\s+landing)?|monterey|mry|santa\s*cruz|santacruz|hmb|half\s*moon\s*bay|princeton)\b[\s:,-]*/i, "");
}
