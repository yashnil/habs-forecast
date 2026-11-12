/**
 * One place to decide what a device's graphics can carry (docs/coastwatch/v12/research/D-motion-feasibility.md §11):
 *   full     WebGL2 on a hardware renderer, more than 4 cores and 4 GB where reported, 4G or better;
 *   reduced  few cores or little memory, 3G or slower, a software renderer, or frames measured slow;
 *   static   no WebGL2, data-saving, offline, a saved copy, or a repeated WebGL context loss: poster only.
 * Reduced motion is separate from the tier (a capable device can ask for stillness). Safe on the server
 * (no window: static) and against every storage or browser API being absent. The renderer is probed
 * once and cached, from one detached canvas whose context is released at once; measured-slow evidence
 * and context losses last the session, so a tier can only come down, never flap back during it.
 */
export type GraphicsTier = "full" | "reduced" | "static";

export type CapabilityInputs = {
  webgl2: boolean;
  softwareRenderer: boolean;
  saved: boolean;
  online: boolean;
  saveData: boolean;
  /** navigator.hardwareConcurrency (absent on some browsers) */
  cores?: number;
  /** navigator.deviceMemory, GB (absent on Safari, so never used alone) */
  memoryGb?: number;
  /** navigator.connection.effectiveType */
  effectiveType?: string;
  slowFrames: boolean;
  contextLosses: number;
};

export type Capability = { tier: GraphicsTier; reducedMotion: boolean; reasons: string[] };

/** Decides a tier from what is known about the device. Pure. */
export function classify(i: CapabilityInputs): { tier: GraphicsTier; reasons: string[] } {
  const barred = [!i.webgl2 && "no WebGL2", i.saved && "saved copy", !i.online && "offline", i.saveData && "data saver", i.contextLosses >= 2 && "repeated context loss"].filter(Boolean) as string[];
  if (barred.length) return { tier: "static", reasons: barred };
  const low = [
    i.softwareRenderer && "software renderer",
    !!i.cores && i.cores <= 4 && "few cores",
    !!i.memoryGb && i.memoryGb <= 4 && "low memory",
    i.effectiveType !== undefined && /^(slow-2g|2g|3g)$/.test(i.effectiveType) && "slow connection",
    i.slowFrames && "frames measured slow",
  ].filter(Boolean) as string[];
  return low.length ? { tier: "reduced", reasons: low } : { tier: "full", reasons: [] };
}

type Nav = Navigator & { deviceMemory?: number; connection?: { saveData?: boolean; effectiveType?: string } };

const SLOW_KEY = "nx-gfx-slow-v1";
const LOSS_KEY = "nx-gfx-losses-v1";

// session evidence, also held here for when sessionStorage is refused
let slow = false;
let losses = 0;

function sessionGet(k: string): string | null {
  try {
    return sessionStorage.getItem(k);
  } catch {
    return null;
  }
}
function sessionSet(k: string, v: string) {
  try {
    sessionStorage.setItem(k, v);
  } catch {
    // held in memory for this page view
  }
}

let probed: { webgl2: boolean; software: boolean } | null = null;

/** The renderer, once per page: whether WebGL2 exists, and whether it is software (a GPU-less fallback). */
export function gpu(): { webgl2: boolean; software: boolean } {
  if (probed) return probed;
  const out = { webgl2: false, software: false };
  try {
    const c = document.createElement("canvas");
    const gl2 = c.getContext("webgl2") as WebGL2RenderingContext | null;
    out.webgl2 = !!gl2;
    const gl = gl2 ?? ((c.getContext("webgl") ?? c.getContext("experimental-webgl")) as WebGLRenderingContext | null);
    if (!gl) out.software = true;
    else {
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      const r = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "";
      out.software = /swiftshader|llvmpipe|software|basic render|softpipe/i.test(r);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    }
  } catch {
    // as no context: the map is not attempted
  }
  probed = out;
  return out;
}

/** Few cores or little memory, where reported (Safari reports no memory). */
export function lowPowerDevice(): boolean {
  if (typeof navigator === "undefined") return false;
  const n = navigator as Nav;
  return !!((n.hardwareConcurrency && n.hardwareConcurrency <= 4) || (n.deviceMemory && n.deviceMemory <= 4));
}

export function reducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

/** What is true of this device right now. */
export function capabilityInputs(): CapabilityInputs {
  const n = (typeof navigator === "undefined" ? {} : navigator) as Nav;
  const g = gpu();
  return {
    webgl2: g.webgl2,
    softwareRenderer: g.software,
    saved: !!(window as unknown as { __NERITIC_SAVED__?: unknown }).__NERITIC_SAVED__,
    online: n.onLine !== false,
    saveData: !!n.connection?.saveData,
    cores: n.hardwareConcurrency || undefined,
    memoryGb: n.deviceMemory || undefined,
    effectiveType: n.connection?.effectiveType,
    slowFrames: slow || sessionGet(SLOW_KEY) === "1",
    contextLosses: Math.max(losses, Number(sessionGet(LOSS_KEY)) || 0),
  };
}

export function capability(): Capability {
  if (typeof window === "undefined") return { tier: "static", reducedMotion: false, reasons: ["server"] };
  const { tier, reasons } = classify(capabilityInputs());
  return { tier, reducedMotion: reducedMotion(), reasons };
}

/** Frames measured slow (the camera's flight median, the currents' frame cost): down once, for the session. */
export function noteSlowFrames() {
  slow = true;
  sessionSet(SLOW_KEY, "1");
}

/** A WebGL context was lost. The first can be recovered from; a second in one session drops to static. */
export function noteContextLoss() {
  losses = Math.max(losses, Number(sessionGet(LOSS_KEY)) || 0) + 1;
  sessionSet(LOSS_KEY, String(losses));
}

/** test hook */
export function resetCapability() {
  probed = null;
  slow = false;
  losses = 0;
}
