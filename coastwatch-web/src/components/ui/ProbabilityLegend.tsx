import type { Palette } from "@/generated/schema";

export function gradientCss(p: Palette): string {
  const [lo, hi] = p.domain;
  const stops = p.stops.map((s) => `${s.color} ${(((s.value - lo) / (hi - lo)) * 100).toFixed(1)}%`);
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

/** Palette colour for a value (linear interpolation between stops, as in the pipeline). */
export function colorAt(p: Palette, v: number): string {
  const [lo, hi] = p.domain;
  const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
  const stops = p.stops;
  let i = 0;
  while (i < stops.length - 2 && stops[i + 1].value < t) i++;
  const a = stops[i];
  const b = stops[i + 1];
  const f = (t - a.value) / Math.max(1e-9, b.value - a.value);
  const ch = (h: string, k: number) => parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16);
  const mix = [0, 1, 2].map((k) => Math.round(ch(a.color, k) + (ch(b.color, k) - ch(a.color, k)) * f));
  return `rgb(${mix.join(",")})`;
}

export function ProbabilityLegend({ palette, threshold }: { palette: Palette; threshold: string | null | undefined }) {
  const ticks = [0, 25, 50, 75, 100];
  return (
    <figure data-testid="probability-legend" className="space-y-1.5" aria-label={`Legend: ${threshold ?? "probability"}, 0 to 100 percent`}>
      <figcaption className="text-[12px] leading-snug text-ink-2">{threshold}</figcaption>
      <div className="h-3 rounded-sm ring-1 ring-hairline" style={{ background: gradientCss(palette) }} />
      <div className="relative h-4 text-[10px] text-ink-3 tabular">
        {ticks.map((t) => (
          <span
            key={t}
            className="absolute -translate-x-1/2 first:translate-x-0 last:-translate-x-full"
            style={{ left: `${t}%` }}
          >
            {t}%
          </span>
        ))}
      </div>
      <div className="flex items-center gap-2 text-[11px] text-ink-3">
        <span
          className="inline-block h-3 w-5 rounded-sm ring-1 ring-hairline-strong"
          style={{ background: "repeating-linear-gradient(45deg, transparent 0 3px, rgba(160,190,225,0.35) 3px 4px)" }}
          aria-hidden
        />
        No value (land, outside model, or not provided near shore)
      </div>
    </figure>
  );
}
