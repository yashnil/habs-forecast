import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { FORECAST_CLASSES, FORECAST_CLASSES_ID, forecastClass } from "@/lib/palette";

const css = readFileSync(path.resolve(__dirname, "../../src/app/globals.css"), "utf8");
const token = (name: string, scope = ":root") => {
  const block = css.slice(css.indexOf(scope === ":root" ? ":root," : `${scope} {`));
  return block.match(new RegExp(`--${name}:\\s*([^;]+);`))?.[1].trim();
};

describe("design tokens", () => {
  it("forecast display classes in CSS match the palette module", () => {
    expect(FORECAST_CLASSES_ID).toBe("cw-probability-classes-v1");
    expect(FORECAST_CLASSES).toHaveLength(10);
    FORECAST_CLASSES.forEach((hex, i) => expect(token(`cw-p${i}`)).toBe(hex));
  });

  it("forecast classes rise monotonically in lightness so order survives greyscale", () => {
    const lum = (h: string) => {
      const c = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255).map((x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
      return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    };
    for (let i = 1; i < 10; i++) expect(lum(FORECAST_CLASSES[i])).toBeGreaterThan(lum(FORECAST_CLASSES[i - 1]));
  });

  it("classes are 10-point display steps; 100 % falls in the top class", () => {
    expect(forecastClass(0)).toBe(0);
    expect(forecastClass(0.0999)).toBe(0);
    expect(forecastClass(0.1)).toBe(1);
    expect(forecastClass(0.755)).toBe(7);
    expect(forecastClass(1)).toBe(9);
  });

  it("reserved semantic colours differ from each other and exist in both themes", () => {
    for (const scope of [":root", ".theme-dark"]) {
      const v = ["cw-official", "cw-forecast", "cw-chl"].map((n) => token(n, scope));
      expect(new Set(v).size).toBe(3);
      v.forEach((x) => expect(x).toBeTruthy());
    }
  });

  it("text tokens on paper meet WCAG AA (4.5:1)", () => {
    const lum = (h: string) => {
      const c = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255).map((x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
      return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    };
    const ratio = (a: string, b: string) => {
      const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
      return (x + 0.05) / (y + 0.05);
    };
    const paper = token("cw-page")!, surface = token("cw-surface")!;
    for (const name of ["cw-ink", "cw-ink-2", "cw-ink-3", "cw-accent", "cw-official", "cw-official-ink", "cw-forecast", "cw-chl", "cw-good", "cw-warning", "cw-serious", "cw-critical", "cw-neutral"]) {
      const fg = token(name)!;
      expect(ratio(fg, paper), `${name} on paper`).toBeGreaterThanOrEqual(4.5);
      expect(ratio(fg, surface), `${name} on surface`).toBeGreaterThanOrEqual(4.5);
    }
    expect(ratio(token("cw-official-ink")!, token("cw-official-bg")!)).toBeGreaterThanOrEqual(4.5);
  });
});
