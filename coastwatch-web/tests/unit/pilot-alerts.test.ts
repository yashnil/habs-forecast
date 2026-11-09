import { describe, expect, it } from "vitest";
import type { PortFact, PortToday } from "@/pilot/contracts";
import { BLOOM_JUMP_PTS, diffFacts, factVersion, snapshotOf, TRIGGERS, type AlertTrigger } from "@/pilot/alerts/diff";
import { MemoryFanoutStore, runFanout, type PushSender } from "@/pilot/alerts/fanout";
import { stubPortToday } from "@/pilot/alerts/factsStub";

const NOW = "2026-11-09T14:00:00.000Z";
const base = stubPortToday("port-592")!;
const all: AlertTrigger[] = [...TRIGGERS];

function withFact(today: PortToday, id: string, patch: Partial<PortFact>): PortToday {
  return { ...today, facts: today.facts.map((f) => (f.id === id ? { ...f, ...patch } : f)) };
}
const snap = (t: PortToday) => snapshotOf(t, "2026-11-08T14:00:00.000Z");

describe("diffFacts (shared by the device inbox and alerts-fanout)", () => {
  it("a first look produces no alerts: alerts start from what was already seen", () => {
    expect(diffFacts(null, base, all, NOW)).toEqual([]);
  });

  it("nothing changed → nothing; a re-check date alone is not a change", () => {
    expect(diffFacts(snap(base), base, all, NOW)).toEqual([]);
    const rechecked = withFact(base, "port-592:status:dungeness_commercial", { as_of: "2026-11-09" });
    expect(diffFacts(snap(base), rechecked, all, NOW)).toEqual([]);
  });

  it("an official status change alerts, names its source and says it is not verified", () => {
    const next = withFact(base, "port-592:status:dungeness_commercial", { severity: "amber", headline: "Commercial Dungeness: opens Jan 1 with a 30-fathom depth limit" });
    const out = diffFacts(snap(base), next, all, NOW);
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ trigger: "status_change", place_id: "port-592", severity: "amber", verified: false });
    expect(out[0].title).toBe("Moss Landing: Commercial Dungeness: opens Jan 1 with a 30-fathom depth limit");
    expect(out[0].body).toContain("Was: “Commercial Dungeness: delayed");
    expect(out[0].body).toContain("not yet verified by a person");
    expect(out[0].key).toBe(`port-592|port-592:status:dungeness_commercial|${factVersion(next.facts[0])}`);
  });

  it("a new toxin result alerts on a new sample date", () => {
    const next = withFact(base, "port-592:test:dungeness", { as_of: "2026-11-05", value: "avg 7.1 ppm (limit 30)" });
    const out = diffFacts(snap(base), next, ["new_toxin_result"], NOW);
    expect(out.map((a) => a.trigger)).toEqual(["new_toxin_result"]);
    expect(out[0].body).toContain("Previous: avg 5.5 ppm");
  });

  it(`bloom risk alerts only on a jump of ≥ ${BLOOM_JUMP_PTS} points or a rise to amber/red, and says it is a model forecast`, () => {
    const id = "port-592:bloom_forecast:charm";
    expect(diffFacts(snap(base), withFact(base, id, { value: "25%" }), all, NOW)).toEqual([]);
    const jump = diffFacts(snap(base), withFact(base, id, { value: "35%" }), all, NOW);
    expect(jump.map((a) => a.trigger)).toEqual(["bloom_risk_jump"]);
    expect(jump[0].body).toContain("model forecast, not a toxin test");
    expect(diffFacts(snap(base), withFact(base, id, { severity: "amber", value: "18%" }), all, NOW)).toHaveLength(1);
    // risk falling is not a "jump"
    const high = withFact(base, id, { severity: "amber", value: "60%" });
    expect(diffFacts(snap(high), withFact(base, id, { value: "5%" }), all, NOW)).toEqual([]);
  });

  it("a chlorophyll outlook never raises an alert (chlorophyll is not toxin)", () => {
    const outlook: PortFact = { id: "port-592:outlook", place_id: "port-592", kind: "bloom_outlook", severity: "neutral", headline: "Chlorophyll outlook steady", basis: "outlook", verified: null, source: { name: "Neritic", url: null }, as_of: "2026-11-08" };
    const a = { ...base, facts: [...base.facts, outlook] };
    const b = { ...base, facts: [...base.facts, { ...outlook, severity: "red" as const, headline: "Chlorophyll outlook high" }] };
    expect(diffFacts(snap(a), b, all, NOW)).toEqual([]);
  });

  it("stale data alerts only when it gets worse", () => {
    const id = "port-592:data_age";
    const stale = withFact(base, id, { severity: "amber", headline: "Neritic's data for this port is 3 days old" });
    expect(diffFacts(snap(base), stale, all, NOW).map((a) => a.trigger)).toEqual(["data_stale"]);
    expect(diffFacts(snap(stale), base, all, NOW)).toEqual([]);
  });

  it("an official record that disappears is 'no longer listed; reason unknown', never 'lifted'", () => {
    const gone = { ...base, facts: base.facts.filter((f) => f.id !== "port-592:status:anchovy_sardine") };
    const out = diffFacts(snap(base), gone, all, NOW);
    expect(out).toHaveLength(1);
    expect(out[0].title).toContain("no longer listed");
    expect(out[0].body).toContain("reason is unknown");
    expect(out[0].body).not.toMatch(/lifted|safe/i);
  });

  it("only the chosen triggers alert; official status first; no alert ever says 'safe'", () => {
    const next = withFact(withFact(base, "port-592:whale:ramp", { headline: "RAMP: depth restriction 30 fathoms in Zone 3" }), "port-592:status:dungeness_commercial", { headline: "Commercial Dungeness: delayed to at least Jan 15" });
    expect(diffFacts(snap(base), next, ["ramp_action"], NOW).map((a) => a.trigger)).toEqual(["ramp_action"]);
    const both = diffFacts(snap(base), next, all, NOW);
    expect(both.map((a) => a.trigger)).toEqual(["status_change", "ramp_action"]);
    for (const a of both) expect(`${a.title} ${a.body}`).not.toMatch(/\bsafe\b/i);
  });
});

describe("runFanout (idempotent server fan-out)", () => {
  const changed = withFact(base, "port-592:status:dungeness_commercial", { headline: "Commercial Dungeness: delayed to at least Jan 15" });
  const setup = () => {
    const s = new MemoryFanoutStore();
    s.subs = [
      { id: "s1", user_id: "u1", place_id: "port-592", triggers: all, channel: "in_app" },
      { id: "s2", user_id: "u2", place_id: "port-592", triggers: ["status_change"], channel: "web_push" },
      { id: "s3", user_id: "u3", place_id: "port-999", triggers: all, channel: "in_app" },
      { id: "s4", user_id: "u4", place_id: "port-592", triggers: ["status_change"], channel: "sms" },
    ];
    s.targets.set("u2", [
      { endpoint: "https://push.example/ok", p256dh: "k", auth: "a" },
      { endpoint: "https://push.example/gone", p256dh: "k", auth: "a" },
    ]);
    return s;
  };
  const sender: PushSender = async (t) => (t.endpoint.endsWith("gone") ? "gone" : "sent");

  it("first run keeps a baseline and sends nothing", async () => {
    const s = setup();
    const r = await runFanout(s, [base], sender, "https://x.test", new Date(NOW));
    expect(r).toMatchObject({ subscriptions: 4, baselined: 3, alerts_written: 0, places_without_facts: ["port-999"] });
    expect(s.state.filter((x) => x.subscription_id === "s1").length).toBe(base.facts.length);
  });

  it("a change writes one inbox alert per subscription, pushes, drops dead endpoints, and never repeats", async () => {
    const s = setup();
    await runFanout(s, [base], sender, "https://x.test", new Date(NOW));
    const r = await runFanout(s, [changed], sender, "https://x.test", new Date(NOW));
    expect(r.alerts_written).toBe(3);
    expect(r.push).toEqual({ sent: 1, gone: 1, failed: 0, skipped_not_configured: 0 });
    expect(r.sms_not_sent).toBe(1);
    expect(s.targets.get("u2")!.map((t) => t.endpoint)).toEqual(["https://push.example/ok"]);
    // rerun with the same data: nothing new
    const again = await runFanout(s, [changed], sender, "https://x.test", new Date(NOW));
    expect(again.alerts_written).toBe(0);
    expect(s.alerts).toHaveLength(3);
  });

  it("a crash before the state was saved cannot duplicate: the claim refuses the second delivery", async () => {
    const s = setup();
    await runFanout(s, [base], sender, "https://x.test", new Date(NOW));
    const saved = s.state.map((x) => ({ ...x }));
    await runFanout(s, [changed], sender, "https://x.test", new Date(NOW));
    s.state = saved; // as if the run died after claiming
    const r = await runFanout(s, [changed], sender, "https://x.test", new Date(NOW));
    expect(r.alerts_written).toBe(0);
    expect(r.duplicates_skipped).toBe(3);
    expect(s.alerts).toHaveLength(3);
  });

  it("without VAPID keys, web-push subscriptions still get the inbox alert and the push is counted as not configured", async () => {
    const s = setup();
    await runFanout(s, [base], null, "https://x.test", new Date(NOW));
    const r = await runFanout(s, [changed], null, "https://x.test", new Date(NOW));
    expect(r.alerts_written).toBe(3);
    expect(r.push.skipped_not_configured).toBe(1);
    expect(r.push.sent).toBe(0);
  });
});
