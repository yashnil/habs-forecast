/** Server URLs for the four fixture datasets. CW_E2E_PORT_BASE moves all four (default 3200),
 *  so parallel checkouts can run the suite side by side. Same numbers as playwright.config.ts. */
export const PORT_BASE = Number(process.env.CW_E2E_PORT_BASE ?? 3200);
export const OK = `http://localhost:${PORT_BASE}`;
export const FAILED = `http://localhost:${PORT_BASE + 1}`;
export const NODATA = `http://localhost:${PORT_BASE + 2}`;
export const M2 = `http://localhost:${PORT_BASE + 3}`;
