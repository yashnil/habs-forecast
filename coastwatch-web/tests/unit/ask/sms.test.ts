import { createHmac } from "node:crypto";
import { describe, expect, it } from "vitest";
import { askDeterministic } from "@/pilot/ask/engine";
import { FIXTURE_NOW } from "@/pilot/ask/fixture";
import { fitSms, gsmLength, homeCommand, keywordOf, segments, SMS_TEXT, toGsm, twiml, twilioSignature, validTwilioRequest } from "@/pilot/ask/sms";
import { handleInboundSms, MemorySmsStore, type SmsDeps } from "@/pilot/ask/smsHandler";
import type { AskRequest } from "@/pilot/ask/types";

const now = new Date(FIXTURE_NOW);
const ask = (r: AskRequest) => askDeterministic(r, { now, mode: "demo" });

describe("Twilio signature (WebCrypto, no SDK)", () => {
  const token = "test-auth-token";
  const url = "https://neritic.example/api/sms";
  const params = { From: "+15555550100", To: "+15555550199", Body: "MOSS crab status", MessageSid: "SM123" };
  const reference = (u: string, p: Record<string, string>) =>
    createHmac("sha1", token)
      .update(u + Object.keys(p).sort().map((k) => k + p[k]).join(""))
      .digest("base64");

  it("matches an independent HMAC-SHA1 of url + sorted params", async () => {
    expect(await twilioSignature(token, url, params)).toBe(reference(url, params));
  });
  it("accepts the right signature and rejects tampering", async () => {
    const sig = reference(url, params);
    expect(await validTwilioRequest(token, url, params, sig)).toBe(true);
    expect(await validTwilioRequest(token, url, { ...params, Body: "STOP" }, sig)).toBe(false);
    expect(await validTwilioRequest(token, `${url}?x=1`, params, sig)).toBe(false);
    expect(await validTwilioRequest("other", url, params, sig)).toBe(false);
    expect(await validTwilioRequest(token, url, params, null)).toBe(false);
  });
});

describe("keywords (CTIA)", () => {
  it.each(["STOP", "stop", " Stop ", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT", "OPTOUT", "REVOKE"])("%s opts out", (w) => expect(keywordOf(w)).toBe("stop"));
  it.each(["START", "UNSTOP", "YES"])("%s opts in", (w) => expect(keywordOf(w)).toBe("start"));
  it.each(["HELP", "info"])("%s is help", (w) => expect(keywordOf(w)).toBe("help"));
  it("a question containing stop is not a keyword", () => expect(keywordOf("when does the crab delay stop")).toBeNull());
  it("program messages fit one or two segments and carry the required words", () => {
    for (const t of [SMS_TEXT.help, SMS_TEXT.start, SMS_TEXT.stop, SMS_TEXT.needPlace]) {
      expect(toGsm(t)).toBe(t);
      expect(gsmLength(t)).toBeLessThanOrEqual(306);
    }
    expect(SMS_TEXT.help).toMatch(/Msg&data rates may apply/);
    expect(SMS_TEXT.help).toMatch(/STOP/);
    expect(SMS_TEXT.start).toMatch(/HELP/);
  });
});

describe("GSM-7 formatting", () => {
  it("replaces typographic characters", () => {
    expect(toGsm("Zones 1–5 — it’s 14.1 °C “ok” …")).toBe('Zones 1-5 - it\'s 14.1C "ok" ...');
  });
  it("cuts at a sentence and keeps the tag", () => {
    const long = Array.from({ length: 20 }, (_, i) => `Sentence number ${i} is here.`).join(" ");
    const out = fitSms(long, "[Src: CDFW 11/5]");
    expect(gsmLength(out)).toBeLessThanOrEqual(306);
    expect(out.endsWith("[Src: CDFW 11/5]")).toBe(true);
    expect(segments(out)).toBeLessThanOrEqual(2);
  });
  it("HOME parses a port", () => {
    expect(homeCommand("HOME MOSS")).toEqual({ place_id: "port-592", label: "Moss Landing" });
    expect(homeCommand("home half moon bay")).toEqual({ place_id: "port-452", label: "Princeton / Half Moon Bay" });
    expect(homeCommand("HOME Atlantis")).toBe("invalid");
    expect(homeCommand("crab status")).toBeNull();
  });
  it("escapes TwiML", () => {
    expect(twiml("a < b & c")).toContain("<Message>a &lt; b &amp; c</Message>");
    expect(twiml(null)).toContain("<Response></Response>");
  });
});

describe("inbound conversation", () => {
  const deps = (over: Partial<SmsDeps> = {}): SmsDeps => ({ store: new MemorySmsStore(), ask, allow: () => true, ...over });
  const phone = "+15555550100";

  it("answers with a place keyword, ≤ 2 segments, with a source tag", async () => {
    const r = await handleInboundSms(phone, "MOSS crab status", deps());
    expect(r).toMatch(/delayed until at least 2027-01-01/);
    expect(r).toMatch(/\[Src: CDFW 11\/5, unverified\]$/);
    expect(gsmLength(r!)).toBeLessThanOrEqual(306);
  });
  it("tags the source of the line shown, and says unverified only for unverified official lines", async () => {
    const r = await handleInboundSms(phone, "MOSS latest crab toxin test", deps());
    expect(r).toMatch(/5\.5 ppm/);
    expect(r).toMatch(/\[Src: CDPH crab 9\/28\]$/);
  });
  it("asks for a port when none is known, then uses the saved home port", async () => {
    const d = deps();
    expect(await handleInboundSms(phone, "crab status", d)).toBe(SMS_TEXT.needPlace);
    expect(await handleInboundSms(phone, "HOME MONTEREY", d)).toBe(SMS_TEXT.homeSet("Monterey"));
    expect(await handleInboundSms(phone, "tide tomorrow", d)).toMatch(/Tide tomorrow at Monterey/);
  });
  it("STOP silences everything until START; HELP always answers", async () => {
    const d = deps();
    expect(await handleInboundSms(phone, "STOP", d)).toBe(SMS_TEXT.stop);
    expect(await handleInboundSms(phone, "MOSS crab status", d)).toBeNull();
    expect(await handleInboundSms(phone, "HELP", d)).toBe(SMS_TEXT.help);
    expect(await handleInboundSms(phone, "START", d)).toBe(SMS_TEXT.start);
    expect(await handleInboundSms(phone, "MOSS crab status", d)).toMatch(/Dungeness/);
  });
  it("stays silent on keywords when Twilio Advanced Opt-Out handles them, but still records the opt-out", async () => {
    const d = deps({ twilioHandlesKeywords: true });
    expect(await handleInboundSms(phone, "STOP", d)).toBeNull();
    expect(await d.store.isOptedOut(phone)).toBe(true);
  });
  it("applies the same safety guard: never 'safe', points to CDPH", async () => {
    const r = await handleInboundSms(phone, "MOSS are mussels safe to eat?", deps());
    expect(r).toMatch(/can't say whether seafood can be eaten/);
    expect(r).toMatch(/CDPH/);
    expect(r).not.toMatch(/\bis safe\b/);
  });
  it("safety question without a port still gets the template", async () => {
    const r = await handleInboundSms(phone, "is it safe to go out tomorrow", deps());
    expect(r).toMatch(/can't say if it's safe to go out/);
  });
  it("rate limits per number", async () => {
    expect(await handleInboundSms(phone, "MOSS tide", deps({ allow: () => false }))).toBe(SMS_TEXT.rateLimited);
  });
  it("out-of-area places are declined", async () => {
    expect(await handleInboundSms(phone, "crab status in Eureka", deps())).toMatch(/only Moss Landing/);
  });
});
