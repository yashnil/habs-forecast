import { describe, expect, it } from "vitest";
import { normalizeQuery, portTier, rankPorts, TIER } from "@/nx/ds/portSearch";

// north to south, as CDFW lists them; regions deliberately left out (they are never matched)
const PORTS = [
  { code: 1, name: "Crescent City", county: "Del Norte", lat: 41.75 },
  { code: 2, name: "Trinidad", county: "Humboldt", lat: 41.06 },
  { code: 3, name: "Eureka", county: "Humboldt", lat: 40.8 },
  { code: 4, name: "Shelter Cove", county: "Humboldt", lat: 40.03 },
  { code: 5, name: "Fort Bragg", county: "Mendocino", lat: 39.44 },
  { code: 6, name: "Bodega Bay", county: "Sonoma", lat: 38.33 },
  { code: 7, name: "San Francisco", county: "San Francisco", lat: 37.8 },
  { code: 8, name: "Princeton-Half Moon Bay", county: "San Mateo", lat: 37.5 },
  { code: 9, name: "Santa Cruz", county: "Santa Cruz", lat: 36.96 },
  { code: 592, name: "Moss Landing", county: "Monterey", lat: 36.8 },
  { code: 11, name: "Monterey", county: "Monterey", lat: 36.6 },
  { code: 12, name: "Morro Bay", county: "San Luis Obispo", lat: 35.37 },
  { code: 13, name: "Port San Luis", county: "San Luis Obispo", lat: 35.17 },
];
const names = (q: string) => rankPorts(PORTS, q).map((p) => p.name);

describe("PortFinder ranking", () => {
  it("puts the port whose name starts with the query first, and only matching ports", () => {
    expect(names("moss")).toEqual(["Moss Landing"]);
    expect(names("eureka")).toEqual(["Eureka"]);
    expect(names("Moss Landing")).toEqual(["Moss Landing"]);
  });

  it("finds nothing for letters that are in no name, county or code", () => {
    expect(names("zzz")).toEqual([]);
    // scattered letters (the old fuzzy match) are not a match
    expect(names("msln")).toEqual([]);
  });

  it("ranks name prefix > word prefix > substring > county", () => {
    // "mo": Monterey, Morro Bay, Moss Landing start with it (north to south among equals); Half Moon is a word prefix
    expect(names("mo")).toEqual(["Moss Landing", "Monterey", "Morro Bay", "Princeton-Half Moon Bay"]);
    expect(portTier(PORTS[7], "half moon")).toBe(TIER.nameWordPrefix);
    expect(portTier(PORTS[4], "ragg")).toBe(TIER.nameSubstring);
    // a county only matches after every name: "monterey" is Monterey the port, then Moss Landing (Monterey County)
    expect(names("monterey")).toEqual(["Monterey", "Moss Landing"]);
    expect(names("humboldt")).toEqual(["Trinidad", "Eureka", "Shelter Cove"]);
  });

  it("matches the CDFW code exactly, as a last resort", () => {
    expect(names("592")).toEqual(["Moss Landing"]);
    expect(names("59")).toEqual([]);
  });

  it("ignores case, accents, punctuation and extra spaces", () => {
    expect(normalizeQuery("  Princeton–Half  Moon ")).toBe("princeton half moon");
    expect(names("MOSS  la")).toEqual(["Moss Landing"]);
    expect(names("half-moon")).toEqual(["Princeton-Half Moon Bay"]);
  });

  it("returns nothing for an empty query (the finder shows the directory instead)", () => {
    expect(names("")).toEqual([]);
    expect(names("   ")).toEqual([]);
  });
});
