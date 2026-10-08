import { describe, expect, it } from "vitest";
import type { OfficialDataset } from "@/generated/official";
import type { SourceStatus } from "@/generated/schema";
import { criticalFlags, officialAt, officialVerification, recordFlags } from "@/lib/official";
import manifestJson from "../fixture-data/v1/manifest.json";
import { readFileSync } from "node:fs";
import path from "node:path";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const base = (): OfficialDataset => JSON.parse(readFileSync(path.join(FIX, (manifestJson as { official_url: string }).official_url), "utf8"));
const at = (iso: string) => new Date(iso);
const verified = (ds: OfficialDataset, reviewedAt = "2026-10-08T18:00:00Z") => {
  ds.registry.review.status = "human_verified";
  ds.registry.review.reviewed_by = "A. Reviewer";
  ds.registry.review.reviewed_at = reviewedAt;
  return ds;
};

describe("official verification status", () => {
  it("the committed transcription is never shown as verified", () => {
    const v = officialVerification(base(), null, at("2026-10-08T20:00:00Z"));
    expect(v.state).toBe("unverified");
    expect(v.reasons.join(" ")).toMatch(/not been checked by a person/);
  });

  it("a fresh human review with unchanged sources is verified", () => {
    expect(officialVerification(verified(base()), null, at("2026-10-09T20:00:00Z")).state).toBe("verified");
  });

  it("loses verified status as the review ages (watcher still running daily)", () => {
    const ds = verified(base());
    ds.watch.forEach((w) => (w.checked_at = "2026-10-12T06:00:00Z"));
    expect(officialVerification(ds, null, at("2026-10-12T12:00:00Z")).state).toBe("aging"); // review 4 days old
    ds.watch.forEach((w) => (w.checked_at = "2026-10-16T06:00:00Z"));
    const old = officialVerification(ds, null, at("2026-10-16T12:00:00Z")); // 8 days
    expect(old.state).toBe("unverified");
    expect(old.reasons.join(" ")).toMatch(/Last review was 8 days ago/);
  });

  it("a changed official page, a new release, or an unreadable page all remove verified status", () => {
    const now = at("2026-10-08T20:00:00Z");
    const changed = verified(base());
    changed.watch[1].matches_review = false;
    expect(officialVerification(changed, null, now).state).toBe("unverified");
    const fresh = verified(base());
    fresh.watch[0].new_items = ["SN26-020"];
    expect(officialVerification(fresh, null, now).reasons.join(" ")).toMatch(/SN26-020/);
    const blocked = verified(base());
    blocked.watch[2] = { ...blocked.watch[2], ok: false, error: "HTTP 403", matches_review: null };
    expect(officialVerification(blocked, null, now).state).toBe("unverified");
  });

  it("contradictory records remove verified status", () => {
    const ds = verified(base());
    ds.conflicts = ["Active 'reopening' record X overlaps active closure Y"];
    expect(officialVerification(ds, null, at("2026-10-08T20:00:00Z")).reasons).toContain("Some records contradict each other.");
  });

  it("a watcher that stopped running removes verified status even if nothing changed", () => {
    const ds = verified(base(), "2026-10-08T18:00:00Z");
    const v = officialVerification(ds, null, at("2026-10-11T20:00:00Z"));
    expect(v.state).not.toBe("verified");
    expect(v.reasons.join(" ")).toMatch(/last checked 3 days ago/);
  });

  it("a failed update and a missing dataset are never verified", () => {
    const failed = { outcome: "failed" } as SourceStatus;
    expect(officialVerification(verified(base()), failed, at("2026-10-08T20:00:00Z")).state).toBe("unverified");
    expect(officialVerification(null, null, at("2026-10-08T20:00:00Z")).state).toBe("unavailable");
  });
});

describe("record flags", () => {
  it("an active record past its expected end is flagged as needing confirmation, never as lifted", () => {
    const r = base().registry.records.find((x) => x.id === "cdph-2026-annual-mussel-quarantine")!;
    expect(criticalFlags(r, at("2026-10-20T12:00:00Z"))).toHaveLength(0);
    const after = criticalFlags(r, at("2026-11-02T12:00:00Z"));
    expect(after[0]).toMatch(/has passed, but no lifting notice is recorded/);
    expect(after[0]).toMatch(/Treat as in effect/);
  });
  it("missing start dates and documented uncertainties are surfaced", () => {
    const r = base().registry.records.find((x) => x.id === "cdfw-rock-crab-commercial-40n")!;
    const f = recordFlags(r, at("2026-10-08T12:00:00Z")).join(" ");
    expect(f).toMatch(/does not state when this closure took effect/);
    expect(f).toMatch(/Offshore extent not stated/);
  });
});

describe("notices at a point", () => {
  const ds = base();
  it("Monterey Bay point: statewide quarantine, Monterey County bivalves, both anchovy notices", () => {
    const ids = officialAt(ds, -121.95, 36.8).map((h) => h.record.id).sort();
    expect(ids).toEqual(
      [
        "cdfw-2026-anchovy-take-restriction-monterey-bay",
        "cdph-2026-annual-mussel-quarantine",
        "cdph-2026-sn26-019-anchovy-central-coast",
      ].sort(),
    );
  });
  it("a point inside the Monterey County outline picks up the county advisory", () => {
    const ids = officialAt(ds, -121.7, 36.4).map((h) => h.record.id);
    expect(ids).toContain("cdph-2026-sn26-018-monterey-bivalves");
  });
  it("outside every notice, only statewide notices apply (and that is not 'open')", () => {
    const hits = officialAt(ds, -117.4, 32.8);
    expect(hits.map((h) => h.record.id)).toEqual(["cdph-2026-annual-mussel-quarantine"]);
  });
});
