import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { MARK_FIELD } from "@/v4/brand/mark.geometry";
import { dateSpan, fieldCaveat, isPastRun } from "@/v4/stage/caption";
import { canRunMap } from "@/v4/stage/fallback";
import { OceanStage, StageCaptionBlock } from "@/v4/stage/OceanStage";
import { SCRIM_STOPS, SCRIM_TEXT_FROM, scrimGradient } from "@/v4/stage/scrim";
import { StagePoster, posterOpacity } from "@/v4/stage/StagePoster";

describe("the stage's honesty", () => {
  it("chlorophyll is never toxin and C-HARM says what it is", () => {
    expect(fieldCaveat("chlorophyll")).toBe("not toxin");
    expect(fieldCaveat("outlook")).toBe("not toxin");
    expect(fieldCaveat("charm")).toBe("probability of conditions, not toxin in seafood");
    expect(fieldCaveat("sst")).toBeNull();
  });
  it("the caption is written with its caveat from the field, not from the caller", () => {
    const html = renderToStaticMarkup(createElement(StageCaptionBlock, { caption: { what: "Chlorophyll", kind: "observed", date: "Oct 6" }, field: "chlorophyll" }));
    expect(html).toContain("not toxin");
    expect(html).toContain("observed");
    expect(html).toContain("Oct 6");
    const past = renderToStaticMarkup(createElement(StageCaptionBlock, { caption: { what: "C-HARM probability", kind: "model", date: "Oct 7", past: true }, field: "charm" }));
    expect(past).toContain("(past run)");
  });
  it("a run whose days are past is a past run; a span reads as dates, not one day", () => {
    expect(isPastRun("2026-10-07", "2026-10-13")).toBe(true);
    expect(isPastRun("2026-10-13", "2026-10-13")).toBe(false);
    expect(dateSpan("2026-10-03", "2026-10-07")).toBe("Oct 3–7");
    expect(dateSpan("2026-09-28", "2026-10-02")).toBe("Sep 28–Oct 2");
    expect(dateSpan(null, "2026-10-07")).toBe("Oct 7");
  });
  it("the map is not attempted on a saved copy, save-data, offline or without WebGL 2", () => {
    const ok = { saved: false, online: true, saveData: false, webgl2: true };
    expect(canRunMap(ok)).toBe(true);
    expect(canRunMap({ ...ok, saved: true })).toBe(false);
    expect(canRunMap({ ...ok, online: false })).toBe(false);
    expect(canRunMap({ ...ok, saveData: true })).toBe(false);
    expect(canRunMap({ ...ok, webgl2: false })).toBe(false);
  });
  it("the poster draws only the real isobath field", () => {
    const html = renderToStaticMarkup(createElement(StagePoster));
    expect(html.match(/<path/g)).toHaveLength(MARK_FIELD.lines.length);
    for (const l of MARK_FIELD.lines) expect(html).toContain(`d="${l.d}"`);
    expect(html).toContain('aria-hidden="true"');
    for (const d of [0, 100, 150, 200, 400, 1000, 2000]) {
      expect(posterOpacity(d)).toBeGreaterThan(0);
      expect(posterOpacity(d)).toBeLessThanOrEqual(0.6);
    }
  });
  it("words and poster are server HTML; the map is not", () => {
    const html = renderToStaticMarkup(createElement(OceanStage, { center: [-121.79, 36.8], zoom: 9, data: null, arrival: "port-592" }, createElement("h1", null, "Moss Landing")));
    expect(html).toContain("Moss Landing");
    expect(html).toContain('data-testid="stage-poster"');
    expect(html).toContain('data-arrival="play"');
    expect(html).not.toContain("maplibregl");
    expect(html).toMatch(/A map of the water/);
  });
  it("the scrim keeps body ink at AA over a pure white pixel, from where text sits", () => {
    const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    const lum = (r: number, g: number, b: number) => 0.2126 * lin(r / 255) + 0.7152 * lin(g / 255) + 0.0722 * lin(b / 255);
    const hex = (h: string) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)] as const;
    const [gr, gg, gb] = hex("#04090e");
    const alpha = (y: number) => {
      for (let i = 1; i < SCRIM_STOPS.length; i++) {
        const [y0, a0] = SCRIM_STOPS[i - 1];
        const [y1, a1] = SCRIM_STOPS[i];
        if (y <= y1) return a0 + ((a1 - a0) * (y - y0)) / (y1 - y0);
      }
      return SCRIM_STOPS[SCRIM_STOPS.length - 1][1];
    };
    for (const ink of ["#eef3f5", "#b3c2cb"]) {
      const [r, g, b] = hex(ink);
      const a = alpha(SCRIM_TEXT_FROM);
      const under = lum(gr * a + 255 * (1 - a), gg * a + 255 * (1 - a), gb * a + 255 * (1 - a));
      const ratio = (lum(r, g, b) + 0.05) / (under + 0.05);
      expect(ratio, ink).toBeGreaterThanOrEqual(4.5);
    }
    expect(SCRIM_STOPS[SCRIM_STOPS.length - 1][1]).toBe(0.85);
    expect(scrimGradient()).toMatch(/85%/);
  });
});
