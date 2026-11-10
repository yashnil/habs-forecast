import { capability, classify } from "@/lib/capability";

/**
 * Whether the live map should be attempted at all: anything but the static tier (`lib/capability`).
 * The poster stays for a saved copy opened offline, a data-saving connection, no connection, no WebGL 2,
 * or a WebGL context lost twice in this session.
 */
export function canRunMap(env: { saved: boolean; online: boolean; saveData: boolean; webgl2: boolean }): boolean {
  return classify({ ...env, softwareRenderer: false, slowFrames: false, contextLosses: 0 }).tier !== "static";
}

export function stageCanRun(): boolean {
  return capability().tier !== "static";
}
