/**
 * Portfolio-preview build (NEXT_PUBLIC_CW_DEMO=1, inlined at build time): the map only,
 * with a first-visit introduction. Bloom Intelligence, Fisheries and observed currents are
 * hidden (next.config.ts redirects their routes). Unset in production builds.
 */
export const DEMO = process.env.NEXT_PUBLIC_CW_DEMO === "1";

/** Sources whose layers the preview hides; their outages are not announced on the map. */
export const DEMO_HIDDEN_SOURCES: readonly string[] = DEMO ? ["hf_radar"] : [];
