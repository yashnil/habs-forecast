import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { capability, capabilityInputs, classify, gpu, lowPowerDevice, noteContextLoss, noteSlowFrames, reducedMotion, resetCapability, type CapabilityInputs } from "@/lib/capability";
import { canRunMap, stageCanRun } from "@/v4/stage/fallback";

const capable: CapabilityInputs = { webgl2: true, softwareRenderer: false, saved: false, online: true, saveData: false, cores: 8, memoryGb: 8, effectiveType: "4g", slowFrames: false, contextLosses: 0 };

describe("classify", () => {
  it("full only for a capable device", () => {
    expect(classify(capable)).toEqual({ tier: "full", reasons: [] });
    // Safari reports no memory and browsers may report no connection: absence is not a mark against the device
    expect(classify({ ...capable, memoryGb: undefined, effectiveType: undefined, cores: undefined }).tier).toBe("full");
  });

  it.each<[Partial<CapabilityInputs>, string]>([
    [{ webgl2: false }, "no WebGL2"],
    [{ saved: true }, "saved copy"],
    [{ online: false }, "offline"],
    [{ saveData: true }, "data saver"],
    [{ contextLosses: 2 }, "repeated context loss"],
  ])("static when %o", (over, why) => {
    const r = classify({ ...capable, ...over });
    expect(r.tier).toBe("static");
    expect(r.reasons).toContain(why);
  });

  it("one context loss can still recover", () => {
    expect(classify({ ...capable, contextLosses: 1 }).tier).toBe("full");
  });

  it.each<[Partial<CapabilityInputs>, string]>([
    [{ softwareRenderer: true }, "software renderer"],
    [{ cores: 4 }, "few cores"],
    [{ memoryGb: 4 }, "low memory"],
    [{ effectiveType: "3g" }, "slow connection"],
    [{ effectiveType: "slow-2g" }, "slow connection"],
    [{ slowFrames: true }, "frames measured slow"],
  ])("reduced when %o", (over, why) => {
    const r = classify({ ...capable, ...over });
    expect(r.tier).toBe("reduced");
    expect(r.reasons).toContain(why);
  });

  it("static outranks reduced", () => {
    expect(classify({ ...capable, saveData: true, cores: 2 }).tier).toBe("static");
  });
});

describe("canRunMap keeps its meaning", () => {
  const ok = { saved: false, online: true, saveData: false, webgl2: true };
  it("only the static conditions stop the map", () => {
    expect(canRunMap(ok)).toBe(true);
    expect(canRunMap({ ...ok, saved: true })).toBe(false);
    expect(canRunMap({ ...ok, online: false })).toBe(false);
    expect(canRunMap({ ...ok, saveData: true })).toBe(false);
    expect(canRunMap({ ...ok, webgl2: false })).toBe(false);
  });
});

describe("on the server", () => {
  it("is static, and never reaches for a browser API", () => {
    resetCapability();
    expect(typeof window).toBe("undefined");
    expect(capability()).toEqual({ tier: "static", reducedMotion: false, reasons: ["server"] });
    expect(reducedMotion()).toBe(false);
    expect(gpu()).toEqual({ webgl2: false, software: false });
  });
});

describe("in a browser", () => {
  const session = new Map<string, string>();
  let contexts = 0;
  const stub = ({ reduced = false, saved = false, renderer = "Apple M2", webgl2 = true, gl = true, nav = {} as Record<string, unknown>, sessionThrows = false } = {}) => {
    contexts = 0;
    const fakeGl = {
      getExtension: (n: string) => (n === "WEBGL_debug_renderer_info" ? { UNMASKED_RENDERER_WEBGL: 1 } : { loseContext: vi.fn() }),
      getParameter: () => renderer,
    };
    vi.stubGlobal("window", { matchMedia: () => ({ matches: reduced }), ...(saved ? { __NERITIC_SAVED__: { at: null } } : {}) });
    vi.stubGlobal("navigator", { onLine: true, hardwareConcurrency: 8, ...nav });
    vi.stubGlobal("document", {
      createElement: () => ({
        getContext: (k: string) => {
          if (k === "webgl2") return webgl2 ? (contexts++, fakeGl) : null;
          return gl ? (contexts++, fakeGl) : null;
        },
      }),
    });
    vi.stubGlobal("sessionStorage", {
      getItem: (k: string) => {
        if (sessionThrows) throw new Error("denied");
        return session.get(k) ?? null;
      },
      setItem: (k: string, v: string) => {
        if (sessionThrows) throw new Error("denied");
        session.set(k, v);
      },
    });
  };
  beforeEach(() => {
    session.clear();
    resetCapability();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("a capable device is full, and stageCanRun agrees", () => {
    stub();
    expect(capability()).toEqual({ tier: "full", reducedMotion: false, reasons: [] });
    expect(stageCanRun()).toBe(true);
  });

  it("reduced motion is reported apart from the tier", () => {
    stub({ reduced: true });
    expect(capability()).toMatchObject({ tier: "full", reducedMotion: true });
  });

  it("probes the renderer once, from one context", () => {
    stub();
    gpu();
    gpu();
    capabilityInputs();
    expect(contexts).toBe(1);
  });

  it("a software renderer without WebGL2 is static; with it, reduced", () => {
    stub({ webgl2: false, renderer: "Google SwiftShader" });
    expect(capability().tier).toBe("static");
    resetCapability();
    stub({ renderer: "Google SwiftShader" });
    expect(capability()).toMatchObject({ tier: "reduced", reasons: ["software renderer"] });
  });

  it("no context at all is static and counted as software", () => {
    stub({ webgl2: false, gl: false });
    expect(capabilityInputs()).toMatchObject({ webgl2: false, softwareRenderer: true });
    expect(stageCanRun()).toBe(false);
  });

  it("reads save-data, offline, a saved copy, cores and memory", () => {
    stub({ nav: { connection: { saveData: true } } });
    expect(capability().reasons).toContain("data saver");
    resetCapability();
    stub({ nav: { onLine: false } });
    expect(capability().reasons).toContain("offline");
    stub({ saved: true });
    expect(capability().reasons).toContain("saved copy");
    resetCapability();
    stub({ nav: { hardwareConcurrency: 2, deviceMemory: 2, connection: { effectiveType: "2g" } } });
    expect(capability()).toMatchObject({ tier: "reduced", reasons: ["few cores", "low memory", "slow connection"] });
    expect(lowPowerDevice()).toBe(true);
  });

  it("measured-slow frames bring the tier down for the session, and it stays down", () => {
    stub();
    noteSlowFrames();
    expect(capability().tier).toBe("reduced");
    noteSlowFrames();
    resetCapability();
    stub();
    session.set("nx-gfx-slow-v1", "1");
    expect(capability().tier).toBe("reduced");
  });

  it("a second context loss in a session drops to static, even across a reload", () => {
    stub();
    noteContextLoss();
    expect(capability().tier).toBe("full");
    noteContextLoss();
    expect(capability()).toMatchObject({ tier: "static", reasons: ["repeated context loss"] });
    resetCapability();
    stub();
    expect(capability().tier).toBe("static");
  });

  it("blocked session storage still counts losses for this page", () => {
    stub({ sessionThrows: true });
    noteContextLoss();
    noteContextLoss();
    expect(capability().tier).toBe("static");
  });
});
