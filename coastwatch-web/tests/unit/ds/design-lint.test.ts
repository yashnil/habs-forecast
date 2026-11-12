import path from "node:path";
import { describe, expect, it } from "vitest";
import { lintRoot, lintSource } from "../../../scripts/design-lint.mjs";

type Finding = { file: string; line: number; rule: string; message: string };
const lint = (file: string, text: string) => lintSource(file, text) as Finding[];
const rules = (file: string, text: string) => lint(file, text).map((f) => f.rule);

const CSS = "src/nx/ds/Thing.module.css";
const TSX = "src/nx/ds/Thing.tsx";
const PAGE = "src/app/(neritic)/today/Page.tsx";

describe("design lint: good samples pass", () => {
  it("a token-only stylesheet is clean", () => {
    const good = `.a { color: var(--ink); background: var(--paper); font: 600 var(--t-16) / 1.4 var(--font-sans);
      padding: var(--s-8) 0; margin: 0 auto; gap: 12px 24px; border-radius: var(--r-0); border-top: var(--rule) solid var(--ink);
      letter-spacing: 0.04em; transition: background-color 120ms linear; }
      .b { font-size: var(--t-14); border-radius: var(--r-1) var(--r-1) 0 0; box-shadow: none; }`;
    expect(lint(CSS, good)).toEqual([]);
  });

  it("the token files may hold colour literals", () => {
    expect(lint("src/nx/ds/tokens.css", ":root { --paper: #fffefa; --halo: rgba(255, 255, 255, 0.92); }")).toEqual([]);
  });

  it("comments, imports, types and directives never count", () => {
    const src = `"use client";
      // #ff0000, 2026-11-10, all clear, box-shadow
      import x from "./go";
      type K = "go" | "safe";
      export const A = () => <p className="x">Next assessment Dec 15–18, subject to change.</p>;`;
    expect(lint(TSX, src)).toEqual([]);
  });

  it("motion presets are allowed where they are defined", () => {
    expect(lint("src/nx/ds/motion.ts", "export const settle = { type: 'spring', stiffness: 260, damping: 30 };")).toEqual([]);
  });

  it("the shipped library and catalog are clean", () => {
    const found = lintRoot(path.resolve(__dirname, "../../..")) as Finding[];
    expect(found.map((f) => `${f.file}:${f.line} ${f.rule}`)).toEqual([]);
  });
});

describe("design lint: each rule catches its tell", () => {
  it("colour literals outside tokens, in CSS and in TSX", () => {
    expect(rules(CSS, ".a { color: #0e1417; }")).toContain("colour-literal");
    expect(rules(CSS, ".a { background: rgba(0,0,0,.2); }")).toContain("colour-literal");
    expect(rules(CSS, ".a { border-color: white; }")).toContain("colour-literal");
    expect(rules(CSS, ".a { --mine: #abc; }")).toContain("colour-literal");
    expect(rules(TSX, `export const A = () => <rect fill="#A83BA6" />;`)).toContain("colour-literal");
    expect(rules(TSX, `export const s = { color: "hsl(10 50% 50%)" };`)).toContain("colour-literal");
  });

  it("text under 14 px", () => {
    expect(rules(CSS, ".a { font-size: 12px; }")).toContain("font-size");
    expect(rules(CSS, ".a { font: 600 13px/1 var(--font-sans); }")).toContain("font-size");
    expect(rules(CSS, ".a { font-size: 0.75rem; }")).toContain("font-size");
    expect(rules(TSX, `export const A = () => <text fontSize={11}>x</text>;`)).toContain("font-size");
    expect(rules(TSX, `export const s = { fontSize: "12px" };`)).toContain("font-size");
  });

  it("shadows, backdrop filters and gradients", () => {
    expect(rules(CSS, ".a { box-shadow: 0 1px 2px var(--ink); }")).toContain("decoration");
    expect(rules(CSS, ".a { backdrop-filter: blur(8px); }")).toContain("decoration");
    expect(rules(CSS, ".a { background: linear-gradient(90deg, var(--ink), var(--paper)); }")).toContain("decoration");
    expect(rules(TSX, `export const s = { boxShadow: "inset 3px 0 0 var(--ink)" };`)).toContain("decoration");
  });

  it("radius not from tokens, and pills", () => {
    expect(rules(CSS, ".a { border-radius: 8px; }")).toContain("radius");
    expect(rules(CSS, ".a { border-radius: 999px; }")).toContain("radius");
    expect(rules(CSS, ".a { border-top-left-radius: 50%; }")).toContain("radius");
    expect(rules(TSX, `export const s = { borderRadius: 4 };`)).toContain("radius");
  });

  it("uppercase, small caps and letter-spaced labels", () => {
    expect(rules(CSS, ".a { text-transform: uppercase; }")).toContain("caps");
    expect(rules(CSS, ".a { font-variant: all-small-caps; }")).toContain("caps");
    expect(rules(CSS, ".a { letter-spacing: 0.12em; }")).toContain("caps");
    expect(rules(TSX, `export const s = { textTransform: "uppercase" };`)).toContain("caps");
  });

  it("emoji", () => {
    expect(rules(TSX, `export const A = () => <p>Seas calm 🌊</p>;`)).toContain("emoji");
    expect(rules(TSX, `export const t = "Alerts ⚠️";`)).toContain("emoji");
  });

  it("forbidden words (T36) in strings and JSX text", () => {
    for (const w of ["Safe to fish", "All clear today", "Go", "a no-go day", "Good day for crab", "Hotspot", "Best spot", "Recommended area", "Expected catch", "Low risk", "AI-powered"]) {
      expect(rules(PAGE, `export const A = () => <p>${w}</p>;`), w).toContain("forbidden-word");
    }
    expect(rules(PAGE, `export const t = "unsafe levels";`)).toContain("forbidden-word");
    // words that merely contain a forbidden word are fine
    expect(rules(PAGE, `export const A = () => <p>From Nov 8, 2 days ago. Safety notices come from CDPH; it goes to CDFW.</p>;`)).toEqual([]);
  });

  it("ISO dates in components, but not in tests", () => {
    expect(rules(TSX, `export const NEXT = "2026-12-15";`)).toContain("iso-date");
    expect(rules(TSX, "export const t = `valid ${a} to 2026-11-05`;")).toContain("iso-date");
    expect(rules("src/nx/ds/thing.test.ts", `const NEXT = "2026-12-15";`)).toEqual([]);
  });

  it("raw transitions over 200 ms, keyframes outside ds/, inline Motion timings", () => {
    expect(rules(CSS, ".a { transition: opacity 400ms ease; }")).toContain("motion");
    expect(rules(CSS, ".a { transition: transform .3s; }")).toContain("motion");
    expect(rules("src/app/(neritic)/x.module.css", "@keyframes spin { to { transform: rotate(1turn); } }")).toContain("motion");
    expect(rules(CSS, "@keyframes ok { to { opacity: 1; } }")).not.toContain("motion");
    expect(rules(TSX, `export const t = { type: "spring", stiffness: 120 };`)).toContain("motion-preset");
    expect(rules(TSX, `export const t = { duration: 0.6, ease: "easeOut" };`)).toContain("motion-preset");
  });

  it("type outside the two faces", () => {
    expect(rules(CSS, ".a { font-family: 'Inter', sans-serif; }")).toContain("font-family");
    expect(rules(CSS, ".a { font: 400 14px/1 ui-monospace, monospace; }")).toContain("font-family");
  });

  it("spacing off the scale", () => {
    expect(rules(CSS, ".a { padding: 7px 10px; }")).toContain("space-scale");
    expect(rules(CSS, ".a { gap: 6px; }")).toContain("space-scale");
  });

  it("allowances need a rule and a reason; with one, the line passes", () => {
    expect(rules(CSS, "/* design-lint-allow space-scale: hairline optical nudge for the rail */\n.a { margin: -1px; }")).toEqual([]);
    expect(rules(CSS, "/* design-lint-allow */\n.a { margin: -1px; }")).toEqual(expect.arrayContaining(["allow-reason", "space-scale"]));
  });

  it("reports file and line", () => {
    const [f] = lint(CSS, ".a {\n  color: var(--ink);\n  box-shadow: 0 0 2px var(--ink);\n}");
    expect(f).toMatchObject({ file: CSS, line: 3, rule: "decoration" });
  });
});

describe("design lint: v13 page rules", () => {
  const PAGE_CSS = "src/nx/coast/coast.module.css";
  const PAGE_TSX = "src/nx/coast/Thing.tsx";
  const all = (file: string, text: string) => lintSource(file, text) as Array<Finding & { level?: string }>;
  it("a page never sizes itself by the margin column (an error); the shell may", () => {
    expect(rules(PAGE_CSS, ".a { width: calc(100vw - var(--margin-w)); }")).toEqual(["margin-width"]);
    expect(all(PAGE_CSS, ".a { width: var(--margin-w); }")[0].level).toBeUndefined();
    expect(lint("src/nx/shell/shell.css", ".a { width: var(--margin-w); }")).toEqual([]);
  });
  it("position: fixed in a page is reported as a warning, and can be excused with a reason", () => {
    const f = all(PAGE_CSS, ".panel { position: fixed; top: 0; }");
    expect(f.map((x) => [x.rule, x.level])).toEqual([["fixed-position", "warn"]]);
    expect(lint(PAGE_CSS, "/* design-lint-allow fixed-position: a coach mark is an overlay */\n.panel { position: fixed; }")).toEqual([]);
    expect(lint("src/nx/ds/Overlay.module.css", ".dialog { position: fixed; }")).toEqual([]);
  });
  it("more than six uses of 14 px text in a page's stylesheet is reported once, with the count", () => {
    const seven = Array.from({ length: 7 }, (_, i) => `.c${i} { font-size: var(--t-14); }`).join("\n");
    const f = all(PAGE_CSS, seven);
    expect(f).toHaveLength(1);
    expect(f[0]).toMatchObject({ rule: "small-text", level: "warn" });
    expect(f[0].message).toContain("7 uses");
    expect(lint(PAGE_CSS, Array.from({ length: 6 }, (_, i) => `.c${i} { font: 400 var(--type-small) / 1.3 var(--font-sans); }`).join("\n"))).toEqual([]);
  });
  it("d3 in a page, a hand-styled button and a second primary are warnings; the design system is exempt", () => {
    const src = `import { scaleLinear } from "d3-scale";
      import { Button } from "@/nx/ds";
      export const A = () => (<div><button type="button">x</button><Button variant="primary">Save</Button><Button variant="primary">Save as new</Button><Button>Other</Button></div>);`;
    const f = all(PAGE_TSX, src);
    expect(f.map((x) => x.rule).sort()).toEqual(["one-primary", "raw-button", "raw-chart"]);
    expect(f.every((x) => x.level === "warn")).toBe(true);
    expect(lint("src/nx/ds/plot/Plot.tsx", src)).toEqual([]);
  });
});
