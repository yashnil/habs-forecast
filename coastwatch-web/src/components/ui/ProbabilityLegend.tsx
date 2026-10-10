import type { Palette } from "@/generated/schema";
import { AGE_COLOURS, gradientCss, SENSOR_COLOURS } from "@/lib/palette";

export { AGE_COLOURS, colorAt, gradientCss } from "@/lib/palette";

export function NoValueSwatch({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-2">
      <span
        className="inline-block h-[11px] w-4 shrink-0 rounded-[2px] border border-[#3a5070]"
        style={{ background: "repeating-linear-gradient(135deg, #0b1d33 0 3px, #556f8f 3px 4px)" }}
        aria-hidden
      />
      {label}
    </span>
  );
}

/** C-HARM probability: ten 10-point display steps (not risk levels). `compact`: one ramp row
 *  with three ticks, sized for the map dock; the caption keeps the threshold and units. */
export function ProbabilityLegend({ palette, threshold, compact = false }: { palette: Palette; threshold: string | null | undefined; compact?: boolean }) {
  const ticks = compact ? [0, 50, 100] : [0, 20, 40, 60, 80, 100];
  const stepped = palette.interpolation === "step";
  return (
    <figure data-testid="probability-legend" className={compact ? "space-y-0.5" : "space-y-1"} aria-label={`Legend: ${threshold ?? "probability"}, 0 to 100 percent${stepped ? " in 10-point display steps" : ""}`}>
      {threshold && <figcaption className={`font-medium leading-snug text-ink ${compact ? "text-[12px]" : "text-[13px] max-sm:text-[12px]"}`}>{threshold}</figcaption>}
      <div className={compact ? "flex items-center gap-3" : ""}>
        <div className="min-w-0 flex-1">
          <div className={`${compact ? "h-2" : "h-2.5"} rounded-[3px]`} style={{ background: gradientCss(palette) }} />
          <div className="relative h-3.5 text-[11px] text-ink-3 tabular" aria-hidden>
            {ticks.map((t) => (
              <span key={t} className="absolute -translate-x-1/2 first:translate-x-0 last:-translate-x-full" style={{ left: `${t}%` }}>
                {t}
                {t === 100 ? "%" : ""}
              </span>
            ))}
          </div>
        </div>
        {compact && <NoValueSwatch label="No value" />}
      </div>
      {compact ? (
        stepped && <p className="text-[11px] leading-tight text-ink-3">10-point colour steps for display, not risk levels.</p>
      ) : (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <NoValueSwatch label="No model value" />
          {stepped && (
            <span className="text-[12px] text-ink-3">
              <span className="sm:hidden">Display steps, not risk levels.</span>
              <span className="max-sm:hidden">10-point colour steps for display, not risk levels.</span>
            </span>
          )}
        </div>
      )}
    </figure>
  );
}

const CHL_TICKS = [0.1, 0.3, 1, 3, 10, 30];

/** Satellite chlorophyll-a on a fixed log scale. */
export function ChlorophyllLegend({ palette, compact = false }: { palette: Palette; compact?: boolean }) {
  const [lo, hi] = palette.domain;
  const ticks = compact ? [0.1, 1, 10] : CHL_TICKS;
  return (
    <figure data-testid="chl-scale-legend" className={compact ? "space-y-0.5" : "space-y-1"} aria-label="Legend: chlorophyll-a, 0.05 to 50 milligrams per cubic metre, log scale">
      <figcaption className={`font-medium leading-snug text-ink ${compact ? "text-[12px]" : "text-[13px]"}`}>
        Chlorophyll-a <span className="font-normal text-ink-3">mg/m³ · log scale{compact ? "" : " · algae biomass, not toxin"}</span>
      </figcaption>
      <div className={compact ? "flex items-center gap-3" : ""}>
        <div className="min-w-0 flex-1">
          <div className={`${compact ? "h-2" : "h-2.5"} rounded-[3px]`} style={{ background: gradientCss(palette) }} />
          <div className="relative h-3.5 text-[11px] text-ink-3 tabular" aria-hidden>
            {ticks.map((t) => (
              <span key={t} className="absolute -translate-x-1/2" style={{ left: `${((Math.log10(t) - lo) / (hi - lo)) * 100}%` }}>
                {t}
              </span>
            ))}
          </div>
        </div>
        {compact && <NoValueSwatch label="No observation" />}
      </div>
      {!compact && <NoValueSwatch label="No observation (cloud, fog, land, or outside the product)" />}
    </figure>
  );
}

export function AgeLegend({ maxDays }: { maxDays: number }) {
  return (
    <figure data-testid="age-legend" className="space-y-1" aria-label="Legend: days since each pixel was observed">
      <figcaption className="text-[13px] font-medium text-ink">Days since each pixel was observed</figcaption>
      <div className="flex gap-0.5">
        {AGE_COLOURS.slice(0, maxDays + 1).map((c, i) => (
          <span key={c} className="flex-1 text-center text-[11px] text-ink-3 tabular">
            <span className="mb-0.5 block h-2.5 rounded-[2px]" style={{ background: c }} aria-hidden />
            {i}
          </span>
        ))}
      </div>
    </figure>
  );
}

export function SensorLegend({ labels }: { labels: string[] }) {
  return (
    <figure data-testid="sensor-legend" className="space-y-1" aria-label="Legend: which sensor each pixel comes from">
      <figcaption className="text-[13px] font-medium text-ink">Sensor shown at each pixel</figcaption>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {labels.map((l, i) => (
          <span key={l} className="inline-flex items-center gap-1.5 text-[12px] text-ink-2">
            <span className="block h-2.5 w-5 rounded-[2px]" style={{ background: SENSOR_COLOURS[i] }} aria-hidden />
            {l}
          </span>
        ))}
      </div>
      <NoValueSwatch label="Neither sensor observed it in 7 days" />
    </figure>
  );
}
