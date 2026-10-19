/**
 * The four marks (design language, PR #43 03-design-language.md): every piece of information is
 * drawn with the mark of how it is known, and the same mark appears on the map, in text and in
 * charts, so the difference survives colour-vision deficiency and greyscale.
 *
 * - official  → a line (rule): latitude limits and area outlines
 * - agency_page → a hollow line: an agency's standing page in Neritic's unreviewed copy, not a notice
 * - measured  → a point that carries its date: filled when sampled in the last 14 days, a ring
 *               when older (age in the shape, never in a colour)
 * - satellite → a pixel (small square)
 * - model     → cells (a 2 × 2 cell glyph)
 * - gap       → hatching: not known
 * - conditions→ a small diamond: context, never a field
 *
 * Marks are decorative (aria-hidden); the words beside them always carry the meaning.
 */
export type MarkKind = "official" | "agency_page" | "measured" | "satellite" | "model" | "gap" | "conditions";

const COLOR: Record<MarkKind, string> = {
  official: "var(--cw-official-line)",
  agency_page: "var(--cw-official-line)",
  measured: "var(--cw-chl)",
  satellite: "var(--cw-chl)",
  model: "var(--cw-forecast)",
  gap: "var(--cw-gap)",
  conditions: "var(--cw-accent)",
};

/** Days after which a measured point is drawn as a ring instead of a filled dot. */
export const MEASURED_RECENT_DAYS = 14;

export function Mark({ kind, aged = false, className = "" }: { kind: MarkKind; aged?: boolean; className?: string }) {
  const c = COLOR[kind];
  const box = kind === "official" || kind === "agency_page" ? { w: 14, h: 12 } : { w: 12, h: 12 };
  return (
    <svg width={box.w} height={box.h} viewBox={`0 0 ${box.w} ${box.h}`} aria-hidden className={`shrink-0 ${className}`} data-mark={kind}>
      {kind === "official" && <rect x="0" y="5" width="14" height="2" rx="1" fill={c} />}
      {kind === "agency_page" && <rect x="0.75" y="4.25" width="12.5" height="3.5" rx="1.25" fill="none" stroke={c} strokeWidth="1.5" />}
      {kind === "measured" && (aged ? <circle cx="6" cy="6" r="3.9" fill="none" stroke={c} strokeWidth="1.6" /> : <circle cx="6" cy="6" r="4.5" fill={c} />)}
      {kind === "satellite" && <rect x="2.5" y="2.5" width="7" height="7" fill={c} />}
      {kind === "model" && (
        <g fill={c}>
          <rect x="1" y="1" width="4.4" height="4.4" />
          <rect x="6.6" y="1" width="4.4" height="4.4" opacity="0.7" />
          <rect x="1" y="6.6" width="4.4" height="4.4" opacity="0.7" />
          <rect x="6.6" y="6.6" width="4.4" height="4.4" />
        </g>
      )}
      {kind === "gap" && (
        <g>
          <rect x="1" y="1" width="10" height="10" fill="none" stroke={c} strokeWidth="1" />
          <path d="M1 7 7 1M3 11 11 3M7 11l4-4" stroke={c} strokeWidth="1" />
        </g>
      )}
      {kind === "conditions" && <rect x="3" y="3" width="6" height="6" transform="rotate(45 6 6)" fill={c} />}
    </svg>
  );
}

/** A kind label: the mark, then the words in sentence case (never tracked capitals). */
export function KindLabel({ kind, aged, children, className = "" }: { kind: MarkKind; aged?: boolean; children: React.ReactNode; className?: string }) {
  const ink: Record<MarkKind, string> = {
    official: "text-official-ink",
    agency_page: "text-official-ink",
    measured: "text-measured",
    satellite: "text-measured",
    model: "text-model-ink",
    gap: "text-ink-3",
    conditions: "text-accent",
  };
  return (
    <p className={`cw-kind ${ink[kind]} ${className}`}>
      <Mark kind={kind} aged={aged} />
      <span>{children}</span>
    </p>
  );
}
