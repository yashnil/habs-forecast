/** The event that opens the bar's port finder (src/nx/shell/PortFinder.tsx), kept apart so pages and tests can use it without the finder's UI. */
export const OPEN_PORT_FINDER = "nx-open-port-finder";
/** Opens the port finder from anywhere (the bar's port control, a page's empty state). */
export const openPortFinder = () => window.dispatchEvent(new Event(OPEN_PORT_FINDER));
