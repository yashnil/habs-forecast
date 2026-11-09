import { createHmac } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FIXTURE_NOW } from "@/pilot/ask/evals/sampleFacts";
import { sampleLoader } from "@/pilot/ask/evals/harness";
import { fixtureServerEnv } from "../fixtures/portToday";
import { entryFor, hashKey, loggerFromEnv, noopLogger, SupabaseAskLogger, type AskLogEntry, type AskLogger } from "@/pilot/ask/logger";
import { meta, serveAsk } from "@/pilot/ask/service";
import { readiness } from "@/pilot/readiness";
import { DEMO_LABEL } from "@/pilot/ask/templates";

const at = new Date(FIXTURE_NOW);

describe("readiness and mode", () => {
  it("no key → demo (deterministic, labelled), with sms not configured", () => {
    expect(readiness("ask", {})).toBe("demo");
    expect(meta({})).toEqual({ readiness: { ask: "demo", sms: "not_configured" }, logging: false, dataset: "published" });
    // the routes pass what the server is reading; the test fixture is labelled
    expect(meta({}, "fixture").dataset).toBe("fixture");
    expect(DEMO_LABEL).toBe("Pilot demo — answers are assembled from Neritic's data without an AI model");
  });
  it("serveAsk without a key answers deterministically and never constructs a client", async () => {
    const a = await serveAsk({ question: "crab status", place: "port-592", first: true }, { env: {}, userKey: "1.2.3.4", channel: "web", at, loader: sampleLoader });
    expect(a.mode).toBe("demo");
    expect(a.model).toBeNull();
    expect(a.notice).toMatch(/not an official source/);
  });
});

describe("logging", () => {
  it("is a no-op by default", () => {
    expect(loggerFromEnv({})).toBe(noopLogger);
    expect(loggerFromEnv({ NERITIC_ASK_LOG: "supabase" })).toBe(noopLogger); // missing credentials
  });
  it("logs question, retrieved ids, answer, mode and model when configured", async () => {
    const rows: AskLogEntry[] = [];
    const logger: AskLogger = { enabled: true, log: async (e) => void rows.push(e) };
    await serveAsk({ question: "latest crab toxin test", place: "port-592" }, { env: {}, userKey: "1.2.3.4", channel: "web", at, loggerOverride: logger, loader: sampleLoader });
    expect(rows).toHaveLength(1);
    const r = rows[0];
    expect(r.question).toBe("latest crab toxin test");
    expect(r.mode).toBe("demo");
    expect(r.model).toBeNull();
    expect(r.retrieved_fact_ids).toContain("test:bma-g-dungeness");
    expect(r.cited_fact_ids).toContain("test:bma-g-dungeness");
    expect(r.answer).toMatch(/5\.5 ppm/);
    expect(r.user_key).toMatch(/^[0-9a-f]{32}$/);
    expect(r.user_key).not.toContain("1.2.3.4");
  });
  it("Supabase logger posts to PostgREST with the service key and never throws", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(new Response(null, { status: 201 }));
    const l = new SupabaseAskLogger("https://x.supabase.co/", "svc", "ask_log", fetchImpl as unknown as typeof fetch);
    const e = entryFor("q", { mode: "demo", model: null, place: null, intent: "help", outcome: "help", citations: [], preface: [], lines: [], guard: { passed: true, removed: 0, codes: [] } } as never, [], "web", null, at);
    await l.log(e);
    expect(fetchImpl.mock.calls[0][0]).toBe("https://x.supabase.co/rest/v1/ask_log");
    expect(fetchImpl.mock.calls[0][1].headers.Authorization).toBe("Bearer svc");
    const broken = new SupabaseAskLogger("https://x", "svc", "ask_log", vi.fn().mockRejectedValue(new Error("down")) as unknown as typeof fetch);
    await expect(broken.log(e)).resolves.toBeUndefined();
  });
  it("hashes user keys with a salt", async () => {
    expect(await hashKey(null, "s")).toBeNull();
    expect(await hashKey("+15555550100", "a")).not.toBe(await hashKey("+15555550100", "b"));
  });
});

// the routes read the app's own fact source (src/pilot/today/portToday.ts): here, the committed fixture
// data and recorded NOAA answers, exactly as the e2e servers do
describe("API routes", () => {
  const server = fixtureServerEnv();
  const asServer = () => Object.entries(server).forEach(([k, v]) => vi.stubEnv(k, v));
  afterEach(() => vi.unstubAllEnvs());

  it("POST /api/ask answers; GET returns readiness; bad input is a 400", async () => {
    asServer();
    vi.stubEnv("ANTHROPIC_API_KEY", "");
    const { GET, POST } = await import("@/app/api/ask/route");
    const res = await POST(new Request("http://x/api/ask", { method: "POST", body: JSON.stringify({ question: "whale rules", place: "port-593" }), headers: { "x-forwarded-for": "9.9.9.9" } }));
    expect(res.status).toBe(200);
    const a = await res.json();
    expect(a.citations[0].fact_id).toBe("whale:ramp:zone-4");
    expect(a.citations[0].today_href).toBe("/today?place=port-593#fact-whale%3Aramp%3Azone-4");
    // the test fixture is never passed off as published data
    expect(a.dataset).toBe("fixture");
    const m = await (await GET(new Request("http://x/api/ask?place=port-593"))).json();
    expect(m.readiness.ask).toBe("demo");
    expect(m.dataset).toBe("fixture");
    expect(m.suggestions.length).toBeGreaterThan(4);
    expect((await POST(new Request("http://x/api/ask", { method: "POST", body: "{}" }))).status).toBe(400);
    expect((await POST(new Request("http://x/api/ask", { method: "POST", body: "nope" }))).status).toBe(400);
  });

  it("POST /api/sms: 503 unconfigured, 403 on a bad signature, TwiML on a good one", async () => {
    const { POST } = await import("@/app/api/sms/route");
    const form = new URLSearchParams({ From: "+15555550100", To: "+15555550199", Body: "MOSS whale rules", MessageSid: "SM1" });
    const mk = (sig: string | null) =>
      new Request("https://neritic.example/api/sms", { method: "POST", body: form, headers: { "content-type": "application/x-www-form-urlencoded", ...(sig ? { "x-twilio-signature": sig } : {}) } });
    vi.stubEnv("TWILIO_ACCOUNT_SID", "");
    vi.stubEnv("TWILIO_AUTH_TOKEN", "");
    expect((await POST(mk(null))).status).toBe(503);

    vi.stubEnv("TWILIO_ACCOUNT_SID", "AC123");
    vi.stubEnv("TWILIO_AUTH_TOKEN", "tok");
    vi.stubEnv("NERITIC_SMS_WEBHOOK_URL", "https://neritic.example/api/sms");
    asServer();
    expect((await POST(mk("bad"))).status).toBe(403);
    const params = Object.fromEntries(form.entries());
    const sig = createHmac("sha1", "tok")
      .update("https://neritic.example/api/sms" + Object.keys(params).sort().map((k) => k + params[k]).join(""))
      .digest("base64");
    const ok = await POST(mk(sig));
    expect(ok.status).toBe(200);
    expect(ok.headers.get("content-type")).toMatch(/text\/xml/);
    const body = await ok.text();
    expect(body).toMatch(/<Message>.*RAMP Zone 4.*\[Src: CDFW RAMP 10\/30, unverified\]<\/Message>/);
  });
});
