/**
 * One entry (docs/coastwatch/v12/design-system/journeys.md, Architecture; research C §6 "One canonical entry").
 * An inline script, run before the first paint of every page in the route group, decides what `/` is:
 *
 *   data-nx-entry="today"   a reader with a saved port (nx-port-v1, or the keys it migrates from): Today
 *                           at that port renders at `/` directly, with no redirect and no flash of home;
 *   data-nx-entry="home"    everyone else: the statement and Find your port on the chart;
 *   data-nx-opening="on"    added for a first visit only (nx-opening-seen-v1 unset), with motion allowed,
 *                           a graphics-capable browser and no deep-link parameters: the opening plays once.
 *
 * No cookie: the port never leaves the device (C §6; the account copy promises device-only storage). The
 * script also applies the chart mode (nx-chart-mode-v1, else the system's dark preference as Dusk; Night is
 * opt-in only) so day, dusk and night are right from the first frame. Any key or pointer skips the opening,
 * even before the app's script has loaded.
 */
export const OPENING_SEEN_KEY = "nx-opening-seen-v1";
export const CHART_MODE_KEY = "nx-chart-mode-v1";
export type ChartMode = "day" | "dusk" | "night";

export const ENTRY_SCRIPT = `(function(){try{var d=document.documentElement,l=window.localStorage,g=function(k){try{return l.getItem(k)}catch(e){return null}};
var m=g("${CHART_MODE_KEY}");if(m!=="day"&&m!=="dusk"&&m!=="night")m=matchMedia("(prefers-color-scheme: dark)").matches?"dusk":"day";d.setAttribute("data-chart-mode",m);
if(location.pathname!=="/")return;
var P=/^port-\\d{1,4}$/,port=null;try{var r=JSON.parse(g("nx-port-v1")||"null");if(r&&r.v===1)port=P.test(r.lastPort||"")?r.lastPort:P.test(r.place||"")?r.place:null;else if(r&&r.v>1)port=null}catch(e){}
if(!port){var t=g("nx-today-place-v1");if(P.test(t||""))port=t;else{var h=g("cw-home-port")||"",x=/"?place"?\\s*:?\\s*"?(port-\\d{1,4})\\b/.exec(h)||/^(port-\\d{1,4})$/.exec(h);if(x)port=x[1]}}
if(port){d.setAttribute("data-nx-entry","today");d.setAttribute("data-nx-port",port);return}
d.setAttribute("data-nx-entry","home");
var q=location.search.slice(1).split("&").filter(function(s){return s&&!/^(utm_[a-z_]+|fbclid|gclid|msclkid|ref|source)=/i.test(s)});
var still=matchMedia("(prefers-reduced-motion: reduce)").matches,weak=!window.WebGL2RenderingContext||self.__NERITIC_SAVED__||navigator.onLine===false||(navigator.connection&&navigator.connection.saveData);
if(g("${OPENING_SEEN_KEY}")==="1"||still||weak||q.length)return;
try{l.setItem("${OPENING_SEEN_KEY}","1")}catch(e){}
d.setAttribute("data-nx-opening","on");setTimeout(function(){if(d.getAttribute("data-nx-opening")==="on")d.setAttribute("data-nx-opening","skip")},5200);
var skip=function(e){if(e&&e.type==="keydown"&&(e.key==="Tab"||e.key==="Shift"||e.metaKey||e.ctrlKey))return;var o=d.getAttribute("data-nx-opening");if(o==="on"||o==="run")d.setAttribute("data-nx-opening","skip");removeEventListener("pointerdown",skip,true);removeEventListener("keydown",skip,true);removeEventListener("wheel",skip,true)};
addEventListener("pointerdown",skip,true);addEventListener("keydown",skip,true);addEventListener("wheel",skip,true)}catch(e){}})()`;
