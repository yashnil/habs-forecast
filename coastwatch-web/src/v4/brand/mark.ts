import { MARK_INDEX_DEPTH, MARK_TIERS, MARK_TILE, MARK_WORD } from "./mark.geometry";

/**
 * Neritic's mark: the head of Monterey Canyon drawn by its own isobaths (100-300 m, every 50 m),
 * turned so the canyon head points up. The nested arches read as depth contours first and as a
 * lowercase n second. The 200 m isobath, the shelf edge that bounds the neritic zone, is the index
 * contour: heavier, in the signal colour.
 *
 * Geometry is generated from NOAA's Coastal Relief Model by scripts/brand/build_mark.py
 * (mark.geometry.ts); this file holds the colours and the one way to draw it. Every surface (the
 * header, the v2 masthead, icons, the sharing cards, the launch screen) draws from here.
 *
 * Three optical sizes, chosen by the drawn size in CSS pixels: `micro` (two lines, up to 20 px:
 * favicons), `small` (three lines, up to 56 px: headers), `full` (five lines: app icons, cards).
 */
export type MarkTier = keyof typeof MARK_TIERS;
export const tierFor = (px: number): MarkTier => (px <= 20 ? "micro" : px <= 56 ? "small" : "full");

export { MARK_TILE, MARK_TIERS, MARK_WORD, MARK_INDEX_DEPTH };

/** 48-unit symbol; the lockup adds a 12-unit gap and the wordmark. */
export const MARK_GAP = 12;
export const LOCKUP_WIDTH = Math.round((48 + MARK_GAP + MARK_WORD.width + 1) * 100) / 100;

export type MarkTone = { tile: string; line: string; index: string; word: string };
export const MARK_TONES = {
  /** light master: reading pages, light grounds (a deep navy tile) */
  light: { tile: "#0e3352", line: "#a9d3df", index: "#5fd0e6", word: "#0e3352" },
  /** dark master: the map's instrument surface (the tile lifted so it holds on the near-black ground) */
  dark: { tile: "#134a6e", line: "#bfe4ec", index: "#7fe3f5", word: "#e8eef2" },
} as const satisfies Record<string, MarkTone>;

/** stroke width of one isobath in a tier */
export const strokeOf = (tier: MarkTier, depth: number) => (depth === MARK_INDEX_DEPTH ? MARK_TIERS[tier].index : MARK_TIERS[tier].minor);

/**
 * The symbol as a standalone SVG string (favicons, app icons, sharing cards).
 * `square`: full-bleed (the OS rounds an app icon). `inset`: the symbol drawn smaller inside a
 * full-bleed square (a maskable icon's safe zone). `mono`: one colour, the lines knocked out of the tile.
 */
export function markSvg({
  tone = "light",
  size = 48,
  tier,
  square = false,
  inset = 0,
  mono,
}: { tone?: keyof typeof MARK_TONES; size?: number; tier?: MarkTier; square?: boolean; inset?: number; mono?: string } = {}): string {
  const t = MARK_TONES[tone];
  const k = tier ?? tierFor(size);
  const id = `nxm${k}${tone}${square ? "s" : ""}${inset}`;
  const shape = square || inset ? "M0 0H48V48H0Z" : MARK_TILE;
  const lines = MARK_TIERS[k].lines
    .map((l) => `<path d="${l.d}" stroke="${mono ? "#000" : l.depth === MARK_INDEX_DEPTH ? t.index : t.line}" stroke-width="${strokeOf(k, l.depth)}"/>`)
    .join("");
  const s = (48 - 2 * inset) / 48;
  const inner = inset ? `<g transform="translate(${inset} ${inset}) scale(${s})">${lines}</g>` : lines;
  const body = mono
    ? `<mask id="${id}m"><path d="${shape}" fill="#fff"/><g fill="none" stroke-linecap="round">${inner}</g></mask><path d="${shape}" fill="${mono}" mask="url(#${id}m)"/>`
    : `<clipPath id="${id}"><path d="${shape}"/></clipPath><g clip-path="url(#${id})"><path d="${shape}" fill="${t.tile}"/><g fill="none" stroke-linecap="round">${inner}</g></g>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="${size}" height="${size}">${body}</svg>`;
}

/** Mark and wordmark as one SVG string (48 units tall). */
export function lockupSvg({ tone = "light", height = 48, mono }: { tone?: keyof typeof MARK_TONES; height?: number; mono?: string } = {}): string {
  const t = MARK_TONES[tone];
  const symbol = markSvg({ tone, size: 48, tier: tierFor(height), mono }).replace(/^<svg[^>]*>|<\/svg>$/g, "");
  const w = (LOCKUP_WIDTH / 48) * height;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${LOCKUP_WIDTH} 48" width="${w}" height="${height}">${symbol}<path transform="translate(${48 + MARK_GAP} ${MARK_WORD.baseline})" d="${MARK_WORD.d}" fill="${mono ?? t.word}"/></svg>`;
}
