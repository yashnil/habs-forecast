import type { LayerArtifact, Manifest, QCCheck } from "@/generated/schema";

export type LineageRow = { label: string; value: string; href?: string; testid?: string };

export type Lineage = {
  title: string;
  /** what kind of product this is, in words */
  kind: string;
  source: LineageRow[];
  time: LineageRow[];
  /** the producer's own processing note (verbatim), e.g. its gap filling; null when not published */
  upstreamNote: string | null;
  inputs: string | null;
  requestUrls: string[];
  checks: QCCheck[];
  failed: number;
  coverage: string | null;
  quantErr: number | null;
  /** the stored values are log10 of the quantity (satellite chlorophyll): the error is in log10 units */
  quantLog: boolean;
  caveats: string[];
  pipeline: LineageRow[];
  citation: string | null;
  licence: string;
};

const KIND: Record<string, string> = {
  official_forecast: "Model output published by its producer (a forecast or nowcast), not an observation",
  observation: "Observation, not a forecast",
};

/**
 * Lineage and quality of one published layer, from the manifest only: where it came from, how its
 * dates were set (an inferred issue date says so), what the producer says it did upstream, every QC
 * check the pipeline ran (failures included), the storage rounding, the caveats and the pipeline
 * run. Nothing is added that the published data do not contain.
 */
export function layerLineage(l: LayerArtifact, m: Pick<Manifest, "pipeline_run_id" | "pipeline_version" | "generated_at">): Lineage {
  const p = l.provenance;
  const up = p.upstream_metadata ?? {};
  const t = l.time;
  const time: LineageRow[] = [];
  if (t.issued_date) time.push({ label: "Run issued", value: `${t.issued_date}${t.issued_date_derived ? ` (inferred: ${t.issued_date_method ?? "not published upstream"})` : ""}`, testid: "lineage-issued" });
  if (t.valid_date) time.push({ label: "Valid", value: `${t.valid_time ?? t.valid_date}${t.lead_days != null ? (t.lead_days === 0 ? " · nowcast" : ` · forecast, ${t.lead_days} day${t.lead_days > 1 ? "s" : ""} ahead`) : ""}` });
  if (t.observed_date) time.push({ label: "Observed", value: t.observed_date });
  const source: LineageRow[] = [
    { label: "Source", value: p.source_name, href: p.source_url },
    ...(p.institution ? [{ label: "Institution", value: p.institution }] : []),
    ...(p.dataset_id ? [{ label: "Dataset", value: p.dataset_id, testid: "lineage-dataset" }] : []),
    ...(p.product_version ? [{ label: "Version", value: p.product_version }] : []),
    { label: "Retrieved", value: p.retrieved_at },
    ...(l.units ? [{ label: "Units", value: l.units }] : []),
  ];
  const inputs = [up.source, [up.satellite, up.platform].filter(Boolean).join(" ")].filter(Boolean).join(" · ") || null;
  const qc = l.qc;
  return {
    title: l.title,
    kind: KIND[l.product_class] ?? l.product_class,
    source,
    time,
    upstreamNote: up.history_first_line ?? null,
    inputs,
    requestUrls: p.request_urls ?? [],
    checks: qc?.checks ?? [],
    failed: (qc?.checks ?? []).filter((c) => !c.passed).length,
    coverage: qc ? `${qc.n_valid.toLocaleString("en-US")} of ${qc.n_cells.toLocaleString("en-US")} cells have a value (${(qc.valid_fraction * 100).toFixed(1)}%)` : null,
    quantErr: l.grid?.max_quantization_error ?? l.vectors?.u_grid.max_quantization_error ?? null,
    quantLog: l.grid?.transform === "log10",
    caveats: l.caveats ?? [],
    pipeline: [
      { label: "Pipeline run", value: m.pipeline_run_id, testid: "lineage-run" },
      { label: "Pipeline version", value: m.pipeline_version },
      { label: "Published", value: m.generated_at },
    ],
    citation: p.citation ?? null,
    licence: p.license,
  };
}
