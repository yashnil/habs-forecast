import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { PortFact } from "@/pilot/contracts";
import { ageText, dataAge, spanWords } from "@/pilot/ask/age";
import { say, support } from "@/pilot/ask/compose";
import { askDeterministic } from "@/pilot/ask/engine";
import { sampleLoader } from "@/pilot/ask/evals/harness";
import { FIXTURE_NOW } from "@/pilot/ask/evals/sampleFacts";
import { check } from "@/pilot/ask/guard";
import { fitSms, gsmLength, segments } from "@/pilot/ask/sms";
import { loadPortToday } from "@/pilot/ask/source";
import { ANSWER_NOTICE, DISCLAIMER } from "@/pilot/ask/templates";
import { UNVERIFIED_LINE } from "@/pilot/ask/unverified";
import { fixtureServerEnv } from "../fixtures/portToday";

/**
 * Ask Neritic's demo answers answer first: one sentence that settles the question, then at most three
 * short supporting lines, each cited; nothing said twice; dates as Today words them; the standing
 * notice short and after the answer. On the app's own facts (the web fixture) and on the eval records.
 */
const env = fixtureServerEnv();
const saved: Record<string, string | undefined> = {};
const NOW = new Date(env.CW_FIXED_NOW);
beforeAll(() => {
  for (const [k, v] of Object.entries(env)) {
    saved[k] = process.env[k];
    process.env[k] = v;
  }
});
afterAll(() => {
  for (const k of Object.keys(env)) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});
const real = (question: string, place: string | null = "port-592", first = true) => askDeterministic({ question, place, first }, { now: NOW, mode: "demo", loader: loadPortToday });
const sample = (question: string, place: string | null = "port-592") => askDeterministic({ question, place, first: true }, { now: new Date(FIXTURE_NOW), mode: "demo", loader: sampleLoader });
const fact = (over: Partial<PortFact>): PortFact => ({ id: "status:dungeness_commercial", place_id: "port-592", kind: "status", severity: "red", headline: "x", basis: "official", verified: false, source: { name: "CDFW", url: null }, as_of: "2026-10-12", ...over });

describe("a yes/no question about one fishery is answered in the first line", () => {
  it("“Is the Monterey crab season open?”: No, the delay and its cause once, then what is next and the recreational rule", async () => {
    const a = await real("Is the Monterey crab season open?", null);
    expect(a.place?.place_id).toBe("port-550");
    expect(a.lines.map((l) => l.text)).toEqual([
      "No — Dungeness crab: commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).",
      "Next CDFW risk assessments (subject to change): December 15-18 and January 13-15.",
      "Dungeness crab: recreational season open since Nov 7 with crab traps prohibited; hoop nets and snares allowed.",
      "Transcribed by Neritic from CDFW and not yet verified by a person; the agency page is the authority.",
    ]);
    expect(a.lines[0]).toMatchObject({ primary: true, cites: ["status:dungeness_commercial"] });
    expect(a.lines[0].source).toMatch(/^CDFW, checked (Oct|Nov) \d+$/);
    expect(a.lines[2]).toMatchObject({ cites: ["status:dungeness_recreational"] });
    // each thing once: the delay, its date and its cause are not repeated from the record's long detail
    const all = a.lines.map((l) => l.text).join(" ");
    for (const once of [/humpback/gi, /Jan(uary)? 1, 2027/g, /delayed/gi]) expect(all.match(once)).toHaveLength(1);
    // every supporting line is cited, and the guard passes the answer as composed
    expect(a.lines.every((l) => l.cites.length > 0)).toBe(true);
    expect(a.guard).toEqual({ passed: true, removed: 0, codes: [] });
    expect(UNVERIFIED_LINE.test(a.lines.at(-1)!.text)).toBe(true);
  });

  it("says No or Yes only where an official closure or delay of the fishery itself settles it", async () => {
    expect((await real("Is commercial salmon open out of Santa Cruz?", "port-593")).lines[0].text).toBe("No — Salmon: the 2026 commercial ocean salmon season has concluded.");
    expect((await real("Is commercial crab closed?")).lines[0].text).toMatch(/^Yes — Dungeness crab: commercial season delayed/);
    // not a yes/no question: the record itself is the answer
    expect((await real("When does the crab season open?")).lines[0].text).toMatch(/^Dungeness crab: commercial season delayed/);
    expect((await real("What's the status of Dungeness crab?")).lines[0].text).toMatch(/^Dungeness crab: commercial season delayed/);
    // a restriction is not a closure, nothing on record is not "open", and a rule on sport harvest or a
    // disagreement between sources is never flattened into a yes or a no
    for (const q of ["Did recreational crab open?", "Is market squid open?", "Is spiny lobster closed?", "Is the mussel season open?", "Is anchovy open?"]) {
      const a = await real(q);
      expect(a.lines[0].text, q).not.toMatch(/^(Yes|No)\b/);
      expect(a.lines.map((l) => l.text).join(" "), q).not.toMatch(/\b(is|are) open\b/i);
    }
    expect((await real("Is market squid open?")).lines.map((l) => l.text)).toEqual(["Market squid: no closure or restriction on record here.", "That is not a guarantee: notices can lag. Check the agencies before you go."]);
    // the eval records (hand-written headlines): the same rule, their own words
    expect((await sample("Is crab season open in Monterey?", null)).lines[0].text).toBe("No — commercial Dungeness crab is delayed until at least 2027-01-01 because of humpback whale entanglements, not domoic acid.");
    expect((await sample("Is commercial salmon open?")).lines[0].text).toBe("Commercial salmon has reopened with trip limits.");
  });

  it("the sector asked about leads and the other sector is left out; a general question shows both", async () => {
    const rec = await real("Did recreational crab open?");
    expect(rec.citations.map((c) => c.fact_id)).toEqual(["status:dungeness_recreational"]);
    const com = await real("Half Moon Bay commercial crab — what's the status?", null);
    expect(com.citations.map((c) => c.fact_id)).not.toContain("status:dungeness_recreational");
    const both = await real("What's the status of Dungeness crab?");
    expect(both.citations.map((c) => c.fact_id)).toEqual(["status:dungeness_commercial", "status:dungeness_recreational"]);
  });

  it("keeps a disagreement between sources and the scope of a sport-harvest rule in the answer", async () => {
    const a = await real("What's the status of mussels?");
    expect(a.lines[0].text).toBe("Shellfish (mussels, clams, oysters): annual quarantine of sport-harvested mussels (sport harvest only).");
    expect(a.lines[1].text).toBe("This is a rule for sport harvest, not a closure of a commercial fishery.");
    expect(a.lines[2].text).toMatch(/^Sources differ: the official-notices registry lists .* so check CDPH before you act\.$/);
    // the notice the row was built from is not quoted a second time
    expect(a.citations.map((c) => c.fact_id)).not.toContain("notice:cdph-2026-annual-mussel-quarantine");
    expect(a.lines.filter((l) => !UNVERIFIED_LINE.test(l.text)).length).toBeLessThanOrEqual(4);
  });
});

describe("what a record adds to its headline (support) and how it is said (say)", () => {
  it("takes the next step and the caveats, one “Next …” only, and leaves a long detail to the Today card", () => {
    const long = "CDFW's October 30, 2026 risk assessment delayed the commercial season in Fishing Zones 1-5 (California/Oregon border to Point Conception) until at least January 1, 2027, because of the number of humpback whale entanglements during the 2026 calendar year; a temporary crab trap prohibition and a Fleet Advisory apply in those zones. Next CDFW risk assessments (subject to change): December 15-18 and January 13-15. Next review Dec 15.";
    expect(support(fact({ detail: long }))).toEqual(["Next CDFW risk assessments (subject to change): December 15-18 and January 13-15."]);
    expect(support(fact({ detail: "The recreational season opened November 7, 2026 to hoop net and loop (snare) gear; crab traps are prohibited from the California/Oregon border to Point Conception (Fishing Zones 1-5) until at least January 1, 2027. Next review Dec 15." }))).toEqual(["Next review Dec 15."]);
    expect(support(fact({ detail: "Neritic's list has no CDFW or CDPH closure or restriction for this fishery here. That is not a guarantee: notices can lag. Check the agencies before you go." }))).toEqual(["That is not a guarantee: notices can lag. Check the agencies before you go."]);
    // a short detail is said whole; a long one without a next step or a caveat is not paraphrased
    expect(support(fact({ detail: "Applies to Zones 1–5, Oregon border to Point Conception. CDFW's next risk assessment is due Dec 15–18." }))).toEqual(["Applies to Zones 1–5, Oregon border to Point Conception. CDFW's next risk assessment is due Dec 15–18."]);
    expect(support(fact({ detail: `${"A sentence about another matter that runs on. ".repeat(6)}`.trim() }))).toEqual([]);
    expect(support(fact({ detail: null }))).toEqual([]);
  });
  it("says a status row as Today's hero does, and any other record in its own words", () => {
    expect(say(fact({ headline: "Dungeness crab (commercial): Commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements)." }))).toBe("Dungeness crab: commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).");
    expect(say(fact({ id: "status:anchovy_sardine", headline: "Anchovy & sardine: Monterey Bay anchovy: no take for human consumption since Sep 11 (domoic acid); bait only." }))).toBe("Monterey Bay anchovy: no take for human consumption since Sep 11 (domoic acid); bait only.");
    expect(say(fact({ headline: "Commercial Dungeness crab is delayed until at least 2027-01-01" }))).toBe("Commercial Dungeness crab is delayed until at least 2027-01-01.");
    expect(say(fact({ id: "notice:x", kind: "notice", headline: "CDPH: Do not eat sport-harvested bivalve shellfish from Monterey County." }))).toBe("CDPH: Do not eat sport-harvested bivalve shellfish from Monterey County.");
  });
});

describe("dates in the reader's terms, as Today words them", () => {
  it("a test is “on Sep 28”, a check is “checked Oct 12”, a run “covered Oct 7–10”: no ISO dates, no hours for a day", async () => {
    const t = await real("What's the latest crab toxin test?");
    const test = t.lines.find((l) => l.primary)!;
    expect(test.text).toBe("Monterey Bay Dungeness tested under the action level on Sep 28: highest 9.8 ppm of 30, all 6 samples under. This is a past lab sample, not a forecast.");
    expect(t.citations.find((c) => c.kind === "test")).toMatchObject({ as_of: "2026-09-28", as_of_words: "Sep 28" });
    expect(t.data_age!.label).toMatch(/^Oldest record here: CDPH .*, Sep 28 \(14 days old\)\. Newest: CDFW \(today\)\.$/);
    // the official status still leads a test question (it outranks any test), in one line
    expect(t.citations[0]).toMatchObject({ basis: "official", fact_id: "status:dungeness_commercial" });
    const b = await real("What's the bloom outlook this week?");
    expect(b.lines.find((l) => l.cites.includes("charm:pda"))!.text).toBe("The latest NOAA C-HARM run (Oct 8) covered Oct 7–10 and is out of date, so Neritic has no current short-range bloom forecast for Moss Landing.");
    expect(b.data_age!.stale).toContain("NOAA C-HARM of Oct 8 is out of date: it covered Oct 7–10.");
    // the outlook is quoted by its plain sentence and what it is; its mg/m³ and held-out skill stay on its Today card
    const o = b.lines.find((l) => l.cites.includes("outlook:chl8d"))!.text;
    expect(o).toBe("Next 8 days (Oct 13–20): chlorophyll near Moss Landing is expected to run higher than usual for this time of year, about 1.6× the typical level. A bloom indicator, not a toxin forecast: chlorophyll is algae of every kind. It is one 8-day average, not a day-by-day forecast, and its typical error is about a factor of 2 either way. A higher-than-usual outlook is a reason to watch the next CDPH test results, not a sign of toxin.");
    for (const a of [t, b]) expect(JSON.stringify([a.lines, a.data_age?.label, a.data_age?.stale, a.caveats])).not.toMatch(/\d{4}-\d{2}-\d{2}|hours old/);
  });
  it("ages a day-dated record in days and a timestamp in hours; spans as Today writes them", () => {
    const now = new Date("2026-10-13T06:00:00Z"); // Oct 12, 11 PM Pacific
    expect(ageText("2026-10-12", now)).toBe("today");
    expect(ageText("2026-10-11", now)).toBe("1 day old");
    expect(ageText("2026-09-28", now)).toBe("14 days old");
    expect(ageText("2026-10-13T03:00:00Z", now)).toBe("3 hours old");
    expect(spanWords("2026-10-07", "2026-10-10", now)).toBe("Oct 7–10");
    expect(spanWords("2026-09-28", "2026-10-03", now)).toBe("Sep 28 – Oct 3");
    expect(spanWords("2026-12-27", "2027-01-03", now)).toBe("Dec 27 – Jan 3, 2027");
    expect(dataAge([fact({ as_of: "2026-10-12" })], now)!.label).toBe("CDFW, Oct 12 (today).");
  });
  it("the guard reads a measurement's date in words as its date, and only the right day", () => {
    const f = fact({ id: "test:x", kind: "test", basis: "measured", verified: null, severity: "info", headline: "Monterey Bay Dungeness tested well under the limit on Sep 28: avg 5.5 ppm, limit 30.", value: "5.5 ppm avg", as_of: "2026-09-28", source: { name: "CDPH", url: null } });
    const now = new Date("2026-11-08T15:00:00Z");
    const codes = (text: string) => check([{ text, cites: ["test:x"] }], [f], now).codes;
    expect(codes("Monterey Bay Dungeness tested well under the limit on Sep 28: avg 5.5 ppm, limit 30.")).toEqual([]);
    expect(codes("Monterey Bay Dungeness averaged 5.5 ppm, limit 30.")).toEqual(["stale_undated"]);
    expect(codes("Monterey Bay Dungeness averaged 5.5 ppm on Sep 2, limit 30.")).toContain("stale_undated");
    expect(codes("Monterey Bay Dungeness averaged 5.5 ppm on Sep 28, 2025, limit 30.")).toContain("stale_undated");
    // a sample from an earlier year must say its year
    const old = { ...f, as_of: "2025-09-28", headline: "Tested on Sep 28, 2025: avg 5.5 ppm, limit 30." };
    expect(check([{ text: "Tested on Sep 28, 2025: avg 5.5 ppm, limit 30.", cites: ["test:x"] }], [old], now).codes).toEqual([]);
    expect(check([{ text: "Tested on Sep 28: avg 5.5 ppm, limit 30.", cites: ["test:x"] }], [old], now).codes).toEqual(["stale_undated"]);
  });
});

describe("the standing notice and the refusals", () => {
  it("is short, comes with the first answer only, and still says the three things it must", async () => {
    const first = await real("Is the Monterey crab season open?", null, true);
    expect(first.notice).toBe(ANSWER_NOTICE);
    expect((await real("What are the whale rules?", "port-592", false)).notice).toBeNull();
    expect(ANSWER_NOTICE.length).toBeLessThan(DISCLAIMER.length * 0.7);
    for (const must of [/not an official source/, /CDFW and CDPH decide/, /take precedence/, /Never use Neritic to decide whether seafood is safe or whether it is safe to go out/]) expect(ANSWER_NOTICE).toMatch(must);
    // the notice is its own field: it is never mixed into the answer's lines
    expect(first.lines.some((l) => /not an official source/.test(l.text))).toBe(false);
  });
  it("“safe to eat” and “safe to go out” keep their fixed templates first, with the records as information only", async () => {
    const eat = await real("Is it safe to eat the crab?");
    expect(eat).toMatchObject({ outcome: "safety", safety: "eat" });
    expect(eat.preface).toEqual(["Neritic doesn't decide whether seafood can be eaten, and it never says that it can.", "CDPH decides that from laboratory tests, and CDFW sets closures. Check them before you eat or sell anything you catch or gather.", "What Neritic has on record is below, for information only."]);
    expect(eat.referrals[0].url).toMatch(/cdph\.ca\.gov/);
    expect(eat.lines.some((l) => /^(yes|no)\b/i.test(l.text))).toBe(false);
    expect(JSON.stringify(eat.lines)).not.toMatch(/\b(is|are) safe\b|safe to eat/i);
    expect(eat.sms).toMatch(/^Neritic can't say whether seafood can be eaten\. CDPH decides/);
    const out = await real("Is it safe to go out tomorrow?");
    expect(out).toMatchObject({ outcome: "safety", safety: "go_out" });
    expect(out.preface[0]).toBe("Neritic can't tell you whether it is safe to go out; that is your call as skipper.");
    expect(out.lines.some((l) => /^(yes|no)\b/i.test(l.text))).toBe(false);
    const fish = await real("Is the season open so I can fish crab?");
    expect(fish).toMatchObject({ outcome: "safety", safety: "fish_legal" });
    expect(fish.lines[0].text).toBe("Dungeness crab: commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).");
  });
});

describe("the same answer on a phone", () => {
  it("leads with the direct answer, fits two segments, and tags the source", async () => {
    const a = await real("Is the Monterey crab season open?", null);
    expect(a.sms).toMatch(/^No - Dungeness crab: commercial season delayed to at least Jan 1, 2027 \(humpback whale entanglements\)\. Next CDFW risk assessments \(subject to change\): December 15-18 and January 13-15\. \[Src: CDFW \d+\/\d+, unverified\]$/);
    expect(gsmLength(a.sms)).toBeLessThanOrEqual(306);
    expect(segments(a.sms)).toBe(2);
    // a short answer is one segment (160 characters)
    const s = await real("Is commercial salmon open out of Santa Cruz?", "port-593");
    expect(s.sms).toMatch(/^No - Salmon: the 2026 commercial ocean salmon season has concluded\. \[Src: CDFW \d+\/\d+, unverified\]$/);
    expect(gsmLength(s.sms)).toBeLessThanOrEqual(160);
    expect(segments(s.sms)).toBe(1);
    // the closing "not yet verified" line is carried by the tag, never spent on characters
    expect(a.sms).not.toMatch(/Transcribed by Neritic/);
  });
  it("cuts at a sentence, never at a decimal point", async () => {
    const body = "Next 8 days (Oct 13-20): chlorophyll near Moss Landing is expected to run higher than usual for this time of year, about 1.6x the typical level. A second sentence that will not fit in what is left of the message at all.";
    const out = fitSms(body, "[Src: C-HARM 10/8]", 190);
    expect(out).toBe("Next 8 days (Oct 13-20): chlorophyll near Moss Landing is expected to run higher than usual for this time of year, about 1.6x the typical level. [Src: C-HARM 10/8]");
    const b = await real("What's the bloom outlook this week?");
    expect(gsmLength(b.sms)).toBeLessThanOrEqual(306);
    expect(b.sms).not.toMatch(/about 1\. \[Src/);
  });
});
