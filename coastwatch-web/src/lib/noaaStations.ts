import NDBC from "@/content/stations/ndbc-ca.json";
import COOPS from "@/content/stations/coops-tide-ca.json";
import type { Station } from "@/lib/conditions";

/**
 * Bundled NOAA station lists (content/stations/README.md). Imported only by the conditions
 * route on the server, so they never enter a page's JavaScript.
 */
export const BUOYS = NDBC as Station[];
export const TIDE_STATIONS = COOPS as Station[];
