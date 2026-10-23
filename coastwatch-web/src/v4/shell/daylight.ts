/**
 * Daylight mode: higher-contrast tokens for bright sun (theme.css, html[data-daylight]). Chosen from the
 * ⋯ menu and kept on this device only. The inline script (layout.tsx) sets the attribute before the
 * first paint, so a reader in daylight mode never sees the darker page flash first.
 */
export const DAYLIGHT_KEY = "nx-daylight-v1";
export const DAYLIGHT_SCRIPT = `(function(){try{if(localStorage.getItem("${DAYLIGHT_KEY}")==="1")document.documentElement.setAttribute("data-daylight","")}catch(e){}})()`;

export function readDaylight(): boolean {
  try {
    return document.documentElement.hasAttribute("data-daylight") || window.localStorage.getItem(DAYLIGHT_KEY) === "1";
  } catch {
    return false;
  }
}

export function setDaylight(on: boolean) {
  const d = document.documentElement;
  if (on) d.setAttribute("data-daylight", "");
  else d.removeAttribute("data-daylight");
  try {
    if (on) window.localStorage.setItem(DAYLIGHT_KEY, "1");
    else window.localStorage.removeItem(DAYLIGHT_KEY);
  } catch {
    // not kept: it applies to this page only
  }
}
