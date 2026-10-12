/**
 * The four kinds of information on CoastWatch, in their reserved colours (design reset §2.2): a
 * user who learns them once can read every page. Used where the product explains itself.
 */
export type Kind = "official" | "model" | "measured" | "historical";

export const KIND: Record<Kind, { label: string; chip: string; dot: string }> = {
  official: { label: "Official", chip: "border-official-line bg-official-bg text-official-ink", dot: "bg-official" },
  model: { label: "Model", chip: "border-model-line bg-model-bg text-model-ink", dot: "bg-model" },
  measured: { label: "Measured", chip: "border-measured-line bg-measured-bg text-measured", dot: "bg-measured" },
  historical: { label: "Historical", chip: "border-history-line bg-history-bg text-history", dot: "bg-history" },
};

export function KindChip({ kind, children }: { kind: Kind; children?: React.ReactNode }) {
  const k = KIND[kind];
  return (
    <span className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[12px] font-semibold uppercase tracking-[0.06em] ${k.chip}`}>
      <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${k.dot}`} />
      {children ?? k.label}
    </span>
  );
}
