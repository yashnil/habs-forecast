/**
 * The loading contour: Monterey Canyon's 200 m isobath, the brand's index line (real bathymetry, the
 * launch screen's field: MARK_FIELD in src/v4/brand/mark.geometry.ts, in tile units), never a made-up
 * curve. While something loads, its soundings join into a line and part again (theme.css .nx-sound,
 * 1.6 s); `loading={false}` (waiting for nothing yet, stale or empty) draws the line still, and so does
 * reduced motion. Callers show it only after real waiting (`nx-late`, 400 ms).
 *
 * The two arms are copied here (about 0.7 KB) rather than importing the whole field into every page's
 * script; a unit test keeps them identical to MARK_FIELD's 200 m line.
 */
export const ISOBATH_200: readonly string[] = ["M21.5 20c-1.5 1-3 1-4 2.5c-1.5 1-2 1-4 5c-1.5 4.5-4.5 15.5-6 19.5c-1.5 4.5-2 4.5-3 6c-1 2-1.5 2.5-3.5 3.5c-2 1.5-7 3-9 4.5c-2 1.5-2 2-3 3.5c-0.5 2-1 2-1 7c0 5.5 2.5 17.5 1.5 25c-1 7.5-5.5 16-7 20c-2 4.5-2 4-3 5.5c-1.5 1.5-3 2.5-4.5 4c-2 1-1.5 1.5-6 3.5c-5 2-16.5 5.5-22 8c-5.5 2-7.5 3-12.5 6c-4.5 3-10 8.5-15 12c-5 4-11.5 7.5-15 10.5c-4 3-5.5 5-8 7", "M21.5 20c1 0.5 1.5 0 2.5 0.5c1.5 0.5 1.5 0 3.5 2.5c1.5 2 5.5 7.5 7 10.5c1.5 3 1.5 3.5 2 7.5c0 4-0.5 11.5 0 15.5c0.5 4 2 6 3 8c1.5 2.5 1.5 2 4.5 5c3 2.5 9.5 7 14 11c4.5 4.5 10 12 13 15.5c3.5 3.5 4 3.5 6.5 5.5c3 2 6 3.5 11.5 5c5.5 1.5 16 3.5 21.5 4c6 0.5 7.5 0 13-1c5-0.5 14-3.5 18-4.5c4-0.5 3.5 0 5 0.5c2 0.5 3.5 1 6 4c3 2.5 8 10.5 10.5 13c2.5 2.5 3 2 4.5 2.5c1.5 0.5 3.5 1 5.5 1.5"];

/** The line's frame in tile units, with a little margin (it reaches x -103.5 to 173, y 20 to 173). */
const FRAME = "-118 6 306 182";

/**
 * `field`: a backdrop that covers its box (dots 2.4 units apart, about 11 px on a desktop);
 * `inline`: a small mark beside loading words, the whole line in view with dots spaced for its size.
 */
export function ContourLoader({ loading = true, size = "field", className }: { loading?: boolean; size?: "field" | "inline"; className?: string }) {
  const inline = size === "inline";
  return (
    <svg className={className} viewBox={FRAME} preserveAspectRatio={inline ? "xMidYMid meet" : "xMidYMid slice"} aria-hidden data-testid="contour-loader" data-state={loading ? "loading" : "still"}>
      <g fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={inline ? 15 : 0.9} style={inline ? ({ "--nx-sound": 30 } as React.CSSProperties) : undefined}>
        {ISOBATH_200.map((d, i) => (
          <path key={i} d={d} className={loading ? "nx-sound" : undefined} />
        ))}
      </g>
    </svg>
  );
}
