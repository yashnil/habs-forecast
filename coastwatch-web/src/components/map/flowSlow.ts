/**
 * The currents' slow-device pause, shared by the layer (which measures) and the page's controls (which say
 * it), apart from the layer so the controls do not pull the simulation into the page's bundle.
 */
let measured = false;
export const FLOW_SLOW_EVENT = "cw-flow-slow";
export const flowTooSlow = () => measured;
export function markFlowTooSlow() {
  measured = true;
  window.dispatchEvent(new Event(FLOW_SLOW_EVENT));
}
