import type Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it, vi } from "vitest";
import { FIXTURE_NOW } from "@/pilot/ask/evals/sampleFacts";
import { costUsd, dailyCap, DEFAULT_DAILY_CAP_USD, RateLimiter, SpendMeter } from "@/pilot/ask/limits";
import { ANSWER_MODEL, askLive, ROUTER_MODEL, SYSTEM_PROMPT, validateLive, type MessagesClient } from "@/pilot/ask/live";
import { fixturePortToday } from "@/pilot/ask/evals/sampleFacts";
import { sampleLoader } from "@/pilot/ask/evals/harness";

/** The live (Claude) path, with a mocked client: no network, no key, no spend. */

const now = new Date(FIXTURE_NOW);
const ctx = { now, mode: "live" as const, loader: sampleLoader };
const facts = fixturePortToday("port-592")!.facts;

function reply(obj: unknown, stop: Anthropic.StopReason = "end_turn"): Anthropic.Message {
  return {
    id: "msg_test",
    type: "message",
    role: "assistant",
    model: ANSWER_MODEL,
    content: [{ type: "text", text: typeof obj === "string" ? obj : JSON.stringify(obj), citations: null }],
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: 3000, output_tokens: 300 },
  } as unknown as Anthropic.Message;
}

function mock(...replies: Anthropic.Message[]) {
  const create = vi.fn<(b: Anthropic.MessageCreateParamsNonStreaming) => Promise<Anthropic.Message>>();
  for (const r of replies) create.mockResolvedValueOnce(r);
  create.mockRejectedValue(new Error("no more mocked replies"));
  return { client: { messages: { create } } as MessagesClient, create };
}

const GOOD = {
  answer: [
    { text: "Commercial Dungeness crab is delayed until at least 2027-01-01 because of humpback whale entanglements. This record is not yet verified by a person.", citations: ["status:dungeness_commercial"] },
    { text: "Recreational Dungeness crab opened 2026-11-07 with crab traps banned.", citations: ["status:dungeness_recreational"] },
  ],
  citations: ["status:dungeness_commercial", "status:dungeness_recreational"],
  covered: true,
};

describe("live answers (mocked Claude)", () => {
  it("sends only the port's facts, the system prompt and a schema whose citation enum is the fact ids", async () => {
    const { client, create } = mock(reply(GOOD));
    const a = await askLive({ question: "What's the status of Dungeness crab?", place: "port-592" }, ctx, { client, meter: new SpendMeter(5) });
    expect(a.mode).toBe("live");
    expect(a.model).toBe(ANSWER_MODEL);
    expect(a.citations.map((c) => c.fact_id)).toEqual(["status:dungeness_commercial", "status:dungeness_recreational"]);
    const body = create.mock.calls[0][0];
    expect(body.model).toBe(ANSWER_MODEL);
    expect(body.system).toBe(SYSTEM_PROMPT);
    const user = String(body.messages[0].content);
    expect(user).toContain("<facts>");
    expect(user).toContain("status:dungeness_commercial");
    expect(user).not.toContain("port-550");
    const schema = body.output_config?.format?.schema as { properties: { citations: { items: { enum: string[] } } } };
    expect(schema.properties.citations.items.enum).toEqual(facts.map((f) => f.id));
    expect(a.data_age?.label).toMatch(/3 days old/);
  });

  it("rejects an answer citing an unknown fact id and regenerates with the reasons", async () => {
    const bad = { ...GOOD, citations: [...GOOD.citations, "status:made-up"], answer: [{ text: "Crab is open.", citations: ["status:made-up"] }] };
    const { client, create } = mock(reply(bad), reply(GOOD));
    const a = await askLive({ question: "crab status", place: "port-592" }, ctx, { client, meter: new SpendMeter(5) });
    expect(create).toHaveBeenCalledTimes(2);
    expect(String(create.mock.calls[1][0].messages.at(-1)!.content)).toMatch(/status:made-up/);
    expect(a.mode).toBe("live");
  });

  it("rejects numbers that are not in the cited facts; two failures fall back to the deterministic answer", async () => {
    const bad = { answer: [{ text: "Crab averaged 4.1 ppm on 2026-09-28; this record is not yet verified by a person.", citations: ["test:bma-g-dungeness"] }], citations: ["test:bma-g-dungeness"], covered: true };
    const { client, create } = mock(reply(bad), reply(bad));
    const a = await askLive({ question: "latest crab toxin test", place: "port-592" }, ctx, { client, meter: new SpendMeter(5) });
    expect(create).toHaveBeenCalledTimes(2);
    expect(a.mode).toBe("live_fallback");
    expect(a.caveats[0]).toMatch(/did not pass Neritic's checks/);
    expect(a.lines.some((l) => /5\.5 ppm/.test(l.text))).toBe(true);
  });

  it("rejects a 'safe' answer", () => {
    const v = validateLive(JSON.stringify({ answer: [{ text: "Seas 5 to 7 ft tomorrow, so it is safe to go out.", citations: ["sea:nws-PZZ535-tomorrow"] }], citations: ["sea:nws-PZZ535-tomorrow"], covered: true }), facts, now);
    expect(v.ok).toBe(false);
    if (!v.ok) expect(v.reasons.join(" ")).toMatch(/safety_claim/);
  });

  it("rejects malformed output and a citation missing from citations[]", () => {
    expect(validateLive("not json", facts, now).ok).toBe(false);
    const v = validateLive(JSON.stringify({ answer: [{ text: "Recreational Dungeness crab opened 2026-11-07. Not yet verified by a person.", citations: ["status:dungeness_recreational"] }], citations: [], covered: true }), facts, now);
    expect(v.ok).toBe(false);
  });

  it("never sends safety questions to the model", async () => {
    const { client, create } = mock();
    const a = await askLive({ question: "Are mussels safe to eat at Moss Landing?", place: null }, ctx, { client, meter: new SpendMeter(5) });
    expect(create).not.toHaveBeenCalled();
    expect(a.outcome).toBe("safety");
    expect(a.preface[0]).toMatch(/doesn't decide whether seafood can be eaten/);
  });

  it("uses the small router model only for questions the deterministic router cannot place", async () => {
    const { client, create } = mock(reply({ intent: "sea" }), reply({ answer: [{ text: "Tide tomorrow at Monterey: high 5.4 ft at 7:12 AM.", citations: ["tide:9413450-tomorrow"] }], citations: ["tide:9413450-tomorrow"], covered: true }));
    const a = await askLive({ question: "will the water be high early on monday", place: "port-592" }, ctx, { client, meter: new SpendMeter(5) });
    expect(create.mock.calls[0][0].model).toBe(ROUTER_MODEL);
    expect(create.mock.calls[1][0].model).toBe(ANSWER_MODEL);
    expect(a.intent).toBe("sea");
  });

  it("off-topic questions stop at the router", async () => {
    const { client, create } = mock(reply({ intent: "off_topic" }));
    const a = await askLive({ question: "write me a poem about boats", place: "port-592" }, ctx, { client, meter: new SpendMeter(5) });
    expect(create).toHaveBeenCalledTimes(1);
    expect(a.outcome).toBe("abstained");
  });

  it("falls back without calling the model once the daily cap is spent", async () => {
    const meter = new SpendMeter(0.001);
    const { client, create } = mock(reply(GOOD));
    const a = await askLive({ question: "crab status", place: "port-592" }, ctx, { client, meter });
    expect(create).not.toHaveBeenCalled();
    expect(a.mode).toBe("live_fallback");
    expect(a.caveats[0]).toMatch(/spend cap/);
  });

  it("meters actual usage", async () => {
    const meter = new SpendMeter(5);
    const { client } = mock(reply(GOOD));
    await askLive({ question: "crab status", place: "port-592" }, ctx, { client, meter });
    expect(meter.spentToday(now)).toBeCloseTo(costUsd(ANSWER_MODEL, 3000, 300), 10);
  });

  it("falls back on a refusal or an API error", async () => {
    const r1 = await askLive({ question: "crab status", place: "port-592" }, ctx, { client: mock(reply("", "refusal")).client, meter: new SpendMeter(5) });
    expect(r1.mode).toBe("live_fallback");
    const failing = { messages: { create: vi.fn().mockRejectedValue(new Error("529 overloaded")) } } as unknown as MessagesClient;
    const r2 = await askLive({ question: "crab status", place: "port-592" }, ctx, { client: failing, meter: new SpendMeter(5) });
    expect(r2.mode).toBe("live_fallback");
    expect(r2.caveats[0]).toMatch(/unavailable/);
  });
});

describe("limits", () => {
  it("rate limiter allows N per window per key", () => {
    const r = new RateLimiter(2, 1000);
    expect([r.take("a", 0), r.take("a", 10), r.take("a", 20), r.take("b", 20), r.take("a", 1500)]).toEqual([true, true, false, true, true]);
  });
  it("spend meter resets each UTC day", () => {
    const m = new SpendMeter(1);
    m.add(0.9, new Date("2026-11-08T10:00:00Z"));
    expect(m.allows(0.2, new Date("2026-11-08T23:00:00Z"))).toBe(false);
    expect(m.allows(0.2, new Date("2026-11-09T01:00:00Z"))).toBe(true);
  });
  it("daily cap comes from NERITIC_ASK_DAILY_USD_CAP with a conservative default", () => {
    expect(dailyCap({})).toBe(DEFAULT_DAILY_CAP_USD);
    expect(dailyCap({ NERITIC_ASK_DAILY_USD_CAP: "4.5" })).toBe(4.5);
    expect(dailyCap({ NERITIC_ASK_DAILY_USD_CAP: "0" })).toBe(0);
    expect(dailyCap({ NERITIC_ASK_DAILY_USD_CAP: "lots" })).toBe(DEFAULT_DAILY_CAP_USD);
  });
});
