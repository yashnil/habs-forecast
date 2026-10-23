import { afterEach, describe, expect, it } from "vitest";
import { aliasMatched, matchRank, rankHits, type SearchHit } from "@/v4/explore/Search";
import { readHomePlace, readHomePort } from "@/v4/state/homePort";

const H = (id: string, label: string, county: string): SearchHit => ({ kind: "harbor", id, label, sub: `Harbour · ${county}` });
const S = (id: string, label: string): SearchHit => ({ kind: "station", id, label, sub: "Shore station · CalHABMAP" });
const ALL: SearchHit[] = [
  H("550", "Monterey", "Monterey"),
  H("592", "Moss Landing", "Monterey"),
  H("452", "Princeton / Half Moon Bay", "San Mateo"),
  H("602", "Avila / Port San Luis", "San Luis Obispo"),
  H("770", "San Pedro", "Los Angeles"),
  H("606", "Morro Bay", "San Luis Obispo"),
  S("HABs-MontereyWharf", "Monterey Wharf"),
  S("HABs-SantaCruzWharf", "Santa Cruz Wharf"),
  { kind: "region", id: "monterey_bay", label: "Monterey Bay", sub: "Region" },
];

describe("place search ranking", () => {
  it("puts the exact name first, then names that start with it, then a word that does, then contains, then county", () => {
    expect(rankHits(ALL, "monterey").map((h) => h.label)).toEqual(["Monterey", "Monterey Wharf", "Monterey Bay", "Moss Landing"]);
    expect(matchRank(ALL[0], "Monterey")).toBe(0);
    expect(matchRank(ALL[6], "monterey")).toBe(1);
    expect(matchRank(ALL[2], "moon")).toBe(2);
    expect(matchRank(ALL[4], "edro")).toBe(3);
    expect(matchRank(ALL[1], "monterey")).toBe(4); // by its county only
    expect(matchRank(ALL[0], "zzz")).toBeNull();
  });
  it("finds a harbour by another name it goes by, and says which", () => {
    expect(rankHits(ALL, "Pillar Point")[0].id).toBe("452");
    expect(aliasMatched(ALL[2], "pillar")).toBe("Pillar Point");
    expect(rankHits(ALL, "port san luis")[0].id).toBe("602");
    // its own name shows the query: no alias note
    expect(aliasMatched(ALL[2], "half moon")).toBeNull();
  });
  it("ignores case and accents, and returns nothing for no match", () => {
    expect(rankHits(ALL, "MÓRRO")[0].label).toBe("Morro Bay");
    expect(rankHits(ALL, "atlantis")).toEqual([]);
  });
});

describe("home port, read tolerantly", () => {
  const set = (v: string | null) => {
    const store = new Map<string, string>(v == null ? [] : [["cw-home-port", v]]);
    (globalThis as unknown as { window: unknown }).window = { localStorage: { getItem: (k: string) => store.get(k) ?? null } };
  };
  afterEach(() => {
    delete (globalThis as unknown as { window?: unknown }).window;
  });
  it("accepts a plain place key or an object with a place", () => {
    set("port-550");
    expect(readHomePlace()).toBe("port-550");
    expect(readHomePort()).toBe(550);
    set(JSON.stringify({ place: "port-201", for: "commercial", species: "dungeness" }));
    expect(readHomePort()).toBe(201);
  });
  it("is no home port when unset, damaged, not a harbour, or storage is unavailable", () => {
    set(null);
    expect(readHomePort()).toBeNull();
    set("{not json");
    expect(readHomePort()).toBeNull();
    set(JSON.stringify({ for: "recreational" }));
    expect(readHomePlace()).toBeNull();
    set("site-12");
    expect(readHomePlace()).toBe("site-12");
    expect(readHomePort()).toBeNull();
    delete (globalThis as unknown as { window?: unknown }).window;
    expect(readHomePort()).toBeNull();
  });
});
