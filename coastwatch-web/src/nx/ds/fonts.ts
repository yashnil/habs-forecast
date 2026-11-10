/**
 * Neritic type (spec section 5): Literata for statements and water names, Public Sans for interface and land.
 * Both SIL OFL 1.1, self-hosted by next/font at build time (no runtime request to Google).
 *
 * Wiring (shell owner): put `fontVariables` on <html> (or any ancestor of a `data-chart-mode` scope). The
 * tokens map `--font-serif` / `--font-sans` onto `--nx-literata` / `--nx-public-sans`.
 *
 * Both are variable fonts: one file per style covers every weight we use (Public Sans 400 to 700; Literata
 * 400 and 500 with the optical-size axis, upright and italic). Spec budget: fonts at most 333 KB, target 200.
 */
import { Literata, Public_Sans } from "next/font/google";

export const literata = Literata({
  variable: "--nx-literata",
  subsets: ["latin"],
  weight: "variable",
  axes: ["opsz"],
  style: ["normal", "italic"],
  display: "swap",
});

export const publicSans = Public_Sans({
  variable: "--nx-public-sans",
  subsets: ["latin"],
  weight: "variable",
  display: "swap",
});

/** Class names that define both font variables. */
export const fontVariables = `${literata.variable} ${publicSans.variable}`;
