/**
 * Neritic type, v14 synthesis: Figtree only (SIL OFL 1.1), self-hosted by next/font at build time (no runtime request
 * to Google). One variable file per style covers the three weights the system uses (500, 650, 750); the italic is the
 * chart's voice for water names and depth labels. Both faces share one family name, so italic text gets the real
 * italic. Together they weigh less than v13's Literata alone.
 *
 * Wiring (shell owner): put `fontVariables` on <html>. tokens.css maps `--font`, `--font-sans` and `--font-serif`
 * (kept for v12/v13 components) onto `--nx-figtree`.
 */
import { Figtree } from "next/font/google";

export const figtree = Figtree({
  variable: "--nx-figtree",
  subsets: ["latin"],
  weight: "variable",
  style: ["normal", "italic"],
  display: "swap",
});

/** Class names that define the font variable. */
export const fontVariables = figtree.variable;
