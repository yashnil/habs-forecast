import { Mark, type MarkKind } from "@/components/ui/Mark";

/**
 * The four kinds of information on Neritic, in their reserved colours (design reset §2.2): a
 * user who learns them once can read every page. Used where the product explains itself.
 */
export type Kind = "official" | "model" | "measured" | "historical";

export const KIND: Record<Kind, { label: string; chip: string; dot: string }> = {
  official: { label: "Official", chip: "border-official-line text-official-ink", dot: "bg-official" },
  model: { label: "Model", chip: "border-model-line bg-model-bg text-model-ink", dot: "bg-model" },
  measured: { label: "Measured", chip: "border-measured-line bg-measured-bg text-measured", dot: "bg-measured" },
  historical: { label: "Historical", chip: "border-history-line bg-history-bg text-history", dot: "bg-history" },
};

const MARK: Record<Kind, MarkKind> = { official: "official", model: "model", measured: "measured", historical: "gap" };
const INK: Record<Kind, string> = { official: "text-official-ink", model: "text-model-ink", measured: "text-measured", historical: "text-history" };

/** A kind label led by its mark, in sentence case (signature P5: no tracked capitals, no tinted chip). */
export function KindChip({ kind, children }: { kind: Kind; children?: React.ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-1.5 whitespace-nowrap text-[13px] font-medium ${INK[kind]}`}>
      <Mark kind={MARK[kind]} />
      {children ?? KIND[kind].label}
    </span>
  );
}
