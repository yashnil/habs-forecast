import { describe, expect, it, vi } from "vitest";

// CSS modules are not processed under vitest; class names echo their key
const css = vi.hoisted(() => ({ default: new Proxy({}, { get: (_t, k) => String(k) }) }));
vi.mock("@/nx/research/research.module.css", () => css);
vi.mock("@/nx/ds/Mark.module.css", () => css);
vi.mock("@/nx/ds/Root.module.css", () => css);
import { HEAD_LINE, layoutStrips, stripHead } from "@/nx/research/RecordStrips";
import type { Strip } from "@/nx/research/model";

const strip = (id: string, extra: Partial<Strip> = {}): Strip =>
  ({ id, v: null, kind: "measured", label: "Particulate domoic acid", unit: "ng/mL in seawater", unitShort: "ng/mL", scale: "log", domain: [0.01, 10], points: [{ t: 1, v: 1, cls: "value" }], ...extra }) as Strip;
const pda = strip("pda", { threshold: { value: 0.5, label: "C-HARM threshold, 500 ng/L" } });

describe("Research strip headers: what the dashes mean is never cut (critic 15)", () => {
  it("gives a threshold strip lines for its unit and the dashes' meaning, as many as the width needs", () => {
    expect(stripHead(strip("chl"), 600)).toBe(26);
    expect(stripHead(pda, 600)).toBe(26 + HEAD_LINE);
    // a phone: the words take two lines
    expect(stripHead(pda, 350)).toBeGreaterThanOrEqual(26 + 2 * HEAD_LINE);
    // a long name with its reading does not fit one phone line either
    const pn = strip("pn", { label: "Pseudo-nitzschia, seriata group", unit: "cells/L, larger cells", threshold: { value: 1e4, label: "C-HARM threshold, 10,000 cells/L (all Pseudo-nitzschia)" } });
    expect(stripHead(pn, 350)).toBeGreaterThanOrEqual(2 * 26 + 2 * HEAD_LINE);
  });

  it("lays each plot under its own header, and a folded strip keeps one line", () => {
    const { rows } = layoutStrips([pda, strip("chl")], 60, new Set(), (s) => stripHead(s, 600));
    expect(rows[0].plotTop - rows[0].top).toBe(26 + HEAD_LINE + 10);
    expect(rows[1].plotTop - rows[1].top).toBe(26 + 4);
    const folded = layoutStrips([pda], 60, new Set(["pda"]), (s) => stripHead(s, 600)).rows[0];
    expect(folded.head).toBe(26);
  });
});
