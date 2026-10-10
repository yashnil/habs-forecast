/**
 * Scientific-safety invariants for user-facing copy and code
 * (docs/coastwatch/05-science-and-safety.md).
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import * as copy from "@/content/copy";
import manifestJson from "../fixture-data/v1/manifest.json";
import type { Manifest } from "@/generated/schema";
import { MODULES, generate } from "../../scripts/gen-types.mjs";

const SRC = path.resolve(__dirname, "../../src");

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = path.join(dir, f);
    return statSync(p).isDirectory() ? files(p) : /\.(tsx?|css)$/.test(f) ? [p] : [];
  });
}

function strings(obj: unknown): string[] {
  if (typeof obj === "string") return [obj];
  if (Array.isArray(obj)) return obj.flatMap(strings);
  if (obj && typeof obj === "object") return Object.values(obj).flatMap(strings);
  return [];
}

const sentences = (t: string) => t.split(/(?<=[.;!?])\s+/).filter(Boolean);
const FORBIDDEN = [/all clear/i, /safe to (eat|fish|harvest)/i, /\bno risk\b/i, /\bgo fishing\b/i, /good (fishing|place to fish)/i, /\brecommended (area|zone|spot)/i, /best (place|spot) to fish/i];

const appCode = files(SRC).filter((f) => !f.includes(`${path.sep}generated${path.sep}`));
const appText = appCode.map((f) => readFileSync(f, "utf8")).join("\n");
const manifest = manifestJson as unknown as Manifest;
const artifactText = manifest.layers.flatMap((l) => [l.title, l.short_title, l.description, l.threshold_text ?? "", ...l.caveats, l.freshness.note]);

describe("copy never labels an area safe or recommends fishing", () => {
  const texts = [...strings(copy), ...artifactText];
  it("'safe' only appears in negated sentences", () => {
    for (const t of texts)
      for (const s of sentences(t))
        if (/(?<!whale[- ])\bsafe(ly)?\b/i.test(s)) expect(s, s).toMatch(/\bnot\b|\bdoes not\b|\bnever\b/i);
  });
  it("contains no forbidden phrases (copy, artifacts, or components)", () => {
    for (const re of FORBIDDEN) {
      for (const t of texts) expect(t, `${re} in copy`).not.toMatch(re);
      expect(appText, `${re} in src`).not.toMatch(re);
    }
  });
});

describe("hierarchy and labelling", () => {
  it("official copy states that a missing notice does not mean open or safe", () => {
    expect(copy.OFFICIAL_STATUS.notTracked).toMatch(/does not mean an area is open/);
    // the registry is transcribed and not human-reviewed; never say a person transcribed or checked it
    expect(copy.OFFICIAL_STATUS.notTracked).toMatch(/has not been checked by a person/);
    expect(copy.OFFICIAL_STATUS.notTracked).not.toMatch(/a person has transcribed/);
    expect(copy.OFFICIAL_STATUS.missingNotOpen).toMatch(/does not mean an area is open or that seafood is safe/);
    expect(copy.OFFICIAL_STATUS.verification.verified).toBe("Verified");
  });
  it("official links point to official agency domains over https", () => {
    for (const l of copy.OFFICIAL_STATUS.links) expect(l.href).toMatch(/^https:\/\/(www\.)?(cdph\.ca\.gov|wildlife\.ca\.gov)\//);
    // the dead link from the old app must not come back
    expect(appText).not.toContain("MarineBiotech.aspx");
  });
  it("forecast copy says it is not a closure decision or a seafood toxin measurement", () => {
    expect(copy.FORECAST_COPY.notA).toMatch(/not a measurement of toxin in seafood/);
    expect(copy.FORECAST_COPY.notA).toMatch(/not a closure decision/);
    expect(copy.FORECAST_COPY.lowNotSafe).toMatch(/does not mean an area is safe/);
  });
  it("chlorophyll is never equated with toxins or fish", () => {
    expect(copy.CHLOROPHYLL_COPY.biomass).toMatch(/does not measure toxins/);
    expect(copy.CHLOROPHYLL_COPY.biomass).toMatch(/does not predict where fish are/);
    for (const l of manifest.layers.filter((x) => x.variable === "chlorophyll_a")) {
      expect(l.product_class).toBe("observation");
      expect(l.caveats.join(" ")).toMatch(/does not measure toxins/);
    }
  });
  it("only C-HARM layers are labelled official forecasts", () => {
    for (const l of manifest.layers) if (l.product_class === "official_forecast") expect(l.group_id).toBe("charm");
  });
  it("primary navigation lists only built experiences (no placeholder entries)", () => {
    expect(copy.EXPERIENCES.map((e) => e.key)).toEqual(["map", "bloom", "fisheries"]);
    for (const e of copy.EXPERIENCES) expect(e.href).toMatch(/^\//);
    expect(JSON.stringify(copy.EXPERIENCES)).not.toMatch(/my coast|upcoming/i);
  });
});

describe("no unsupported claims or relative-risk tiers", () => {
  it("does not repeat the unsupported 8.3% PINN claim", () => {
    expect(appText).not.toMatch(/8\.3\s*%/);
  });
  it("does not rank areas by within-map percentiles", () => {
    expect(appText).not.toMatch(/percentile|tertile|higher than most water/i);
  });
  it("probability palette domain is fixed at 0-1", () => {
    for (const l of manifest.layers.filter((x) => x.palette && x.units.startsWith("probability"))) expect(l.palette!.domain).toEqual([0, 1]);
  });
  it("every palette is fixed: one palette id never has two domains (ranges never follow the data)", () => {
    const byId = new Map<string, string>();
    for (const l of manifest.layers.filter((x) => x.palette)) {
      const key = JSON.stringify([l.palette!.domain, l.palette!.stops]);
      expect(byId.get(l.palette!.id) ?? key).toBe(key);
      byId.set(l.palette!.id, key);
    }
    const chl = manifest.layers.find((l) => l.layer_id === "olci300_chl_latest");
    expect(chl?.palette?.scale).toBe("log10");
    expect(chl?.palette?.domain.map((d) => Math.round(10 ** d * 100) / 100)).toEqual([0.05, 50]);
  });
});

describe("generated types", () => {
  it("src/generated/*.ts are up to date with schemas/v1", async () => {
    for (const [mod, names] of Object.entries(MODULES)) {
      const current = readFileSync(path.join(SRC, "generated", `${mod}.ts`), "utf8");
      expect(await generate(names), mod).toBe(current);
    }
  });
});

describe("published research is cited accurately", () => {
  it("names the paper and DOI, says CoastWatch does not run its models, and claims no review of the app", () => {
    expect(copy.RESEARCH.doi).toBe("https://doi.org/10.33422/ccgconf.v2i2.1619");
    expect(copy.RESEARCH.title).toBe("Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System");
    expect(copy.RESEARCH.relation).toMatch(/does not run those research models/);
    expect(copy.RESEARCH.relation).toMatch(/has not been reviewed by an independent HAB scientist/);
    // the same-run evaluation does not show the PINN beating the ConvLSTM
    expect(copy.RESEARCH.relation).not.toMatch(/outperform|beats|better than|improves on/i);
  });
});

