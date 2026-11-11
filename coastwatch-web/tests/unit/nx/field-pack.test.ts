import { describe, expect, it } from "vitest";
import { gridFiles } from "@/nx/explore/chunks";
import { packWords, publicRefs, type FieldRecord } from "@/nx/offline/pack";

const rec: FieldRecord = {
  v: 1,
  port: { key: "port-592", name: "Moss Landing" },
  savedAt: "2026-10-13T05:00:00Z",
  pages: [
    { id: "today", label: "Today", path: "/today?place=port-592", ok: true },
    { id: "coast", label: "My Coast", path: "/coast?place=port-592", ok: true },
    { id: "explore", label: "Explore", path: "/explore?place=port-592", ok: true },
    { id: "research", label: "Research", path: "/research?place=port-592", ok: false },
  ],
  frames: [],
  station: null,
  files: 10,
  bytes: 1000,
  missing: ["Research (the page did not load)"],
};

describe("field pack", () => {
  it("names only the pages it holds, and what waits for a connection on each page", () => {
    const w = packWords(rec, "explore");
    expect(w.saved).toBe("Saved for Moss Landing: Today, My Coast and Explore's latest frames.");
    expect(w.notSaved).toMatch(/latest frame is saved, as a still picture/);
    expect(packWords(null, "coast").saved).toBeNull();
    expect(packWords(null, "coast").notSaved).toMatch(/worksheet works as usual/);
  });

  it("keeps the icons, manifest and images a page names, never another page", () => {
    const html = '<link rel="icon" href="/favicon.ico?v=2"/><link rel="manifest" href="/manifest.webmanifest"/><a href="/about">About</a><img src="/chart/opening-canyon.webp"/><a href="//evil.example/x.png">x</a>';
    expect(publicRefs(html).sort()).toEqual(["/chart/opening-canyon.webp", "/favicon.ico", "/manifest.webmanifest"]);
  });

  it("saves a chunked grid only by the chunks around the port, a small grid whole", () => {
    const base = { width: 4000, height: 2000, lat_first: 32, lat_step: 0.005, lon_first: -125, lon_step: 0.005, scale_factor: 1, add_offset: 0, nodata: 65535, transform: "none" as const };
    const chunked = { ...base, url: null, chunks: { rows: 256, cols: 256, url_template: "g/{row}_{col}.gz", present: ["0_0", "3_1", "4_2", "7_15"] } };
    // a box over Monterey Bay: chunk rows 3–4, columns 1–2
    expect(gridFiles(chunked, [-122.5, 36.4, -121.5, 37.2])).toEqual(["g/3_1.gz", "g/4_2.gz"]);
    expect(gridFiles({ ...base, url: "g/whole.gz", chunks: null }, [-122.5, 36.4, -121.5, 37.2])).toEqual(["g/whole.gz"]);
    expect(gridFiles(null, null)).toEqual([]);
  });
});
