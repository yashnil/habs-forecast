/**
 * Whether the live map should be attempted at all. Pure given its inputs; `stageCanRun` reads them.
 * The poster stays for a saved copy opened offline, a data-saving connection, no connection, or no WebGL 2.
 */
export function canRunMap(env: { saved: boolean; online: boolean; saveData: boolean; webgl2: boolean }): boolean {
  return !env.saved && env.online && !env.saveData && env.webgl2;
}

function hasWebgl2(): boolean {
  try {
    return !!document.createElement("canvas").getContext("webgl2");
  } catch {
    return false;
  }
}

export function stageCanRun(): boolean {
  const n = navigator as Navigator & { connection?: { saveData?: boolean } };
  return canRunMap({ saved: !!(window as unknown as { __NERITIC_SAVED__?: unknown }).__NERITIC_SAVED__, online: n.onLine !== false, saveData: !!n.connection?.saveData, webgl2: hasWebgl2() });
}
