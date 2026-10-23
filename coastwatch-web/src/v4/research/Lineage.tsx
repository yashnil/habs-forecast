"use client";

import type { LayerArtifact, Manifest } from "@/generated/schema";
import { Icon } from "@/v4/ui/Icon";
import { layerLineage, type LineageRow } from "./layerLineage";

function Rows({ rows }: { rows: LineageRow[] }) {
  return (
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[13.5px]">
      {rows.map((r) => (
        <div key={r.label} className="contents">
          <dt className="text-n-ink-3">{r.label}</dt>
          <dd className="min-w-0 break-words" data-testid={r.testid}>
            {r.href ? (
              <a href={r.href} target="_blank" rel="noreferrer" className="nx-focus inline-flex items-center gap-1 font-semibold text-n-accent underline-offset-2 hover:underline">
                {r.value} <Icon name="external" className="h-3.5 w-3.5 shrink-0" />
              </a>
            ) : (
              r.value
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Lineage and QC of the layer on the Research map (manifest only; see lineage.ts). Failed checks
 * are listed as failed, first; the producer's own processing note is quoted, never paraphrased.
 */
export function Lineage({ layer, manifest }: { layer: LayerArtifact; manifest: Manifest }) {
  const L = layerLineage(layer, manifest);
  const checks = [...L.checks].sort((a, b) => Number(a.passed) - Number(b.passed));
  return (
    <section className="nx-deck grid gap-5 p-4 sm:p-5 lg:grid-cols-2" data-testid="research-lineage" aria-labelledby="lineage-h">
      <div className="flex min-w-0 flex-col gap-4">
        <div>
          <h2 id="lineage-h" className="nx-eyebrow">
            Lineage and quality · this layer
          </h2>
          <p className="mt-1 text-[15px] font-semibold leading-snug">{L.title}</p>
          <p className="text-[13px] text-n-ink-2">{L.kind}</p>
        </div>
        <Rows rows={[...L.source, ...L.time]} />
        {(L.upstreamNote || L.inputs) && (
          <div className="rounded-lg border border-n-line bg-n-surface-2 p-3 text-[13.5px] leading-snug" data-testid="lineage-upstream">
            <p className="nx-eyebrow mb-1">Upstream, as stated by the producer</p>
            {L.inputs && <p className="text-n-ink-2">Inputs: {L.inputs}</p>}
            {L.upstreamNote && <p className="mt-1">&ldquo;{L.upstreamNote}&rdquo;</p>}
            <p className="mt-1 text-[12.5px] text-n-ink-3">Any gap filling is the producer&rsquo;s, before the model runs. Neritic fills no gaps and interpolates nothing.</p>
          </div>
        )}
        {!!L.requestUrls.length && (
          <details className="text-[13px]">
            <summary className="nx-focus min-h-11 cursor-pointer py-2 font-semibold text-n-accent">The {L.requestUrls.length === 1 ? "request" : `${L.requestUrls.length} requests`} the pipeline made</summary>
            <ul className="flex flex-col gap-1.5">
              {L.requestUrls.map((u) => (
                <li key={u} className="nx-data break-all text-[12px] text-n-ink-2">
                  {u}
                </li>
              ))}
            </ul>
          </details>
        )}
        <Rows rows={L.pipeline} />
        <div className="flex flex-col gap-2 text-[13px] leading-snug text-n-ink-2">
          {L.citation && (
            <p>
              <b className="font-semibold text-n-ink">Citation.</b> {L.citation}
            </p>
          )}
          <p data-testid="lineage-licence">
            <b className="font-semibold text-n-ink">Licence.</b> {L.licence}
          </p>
        </div>
      </div>
      <div className="flex min-w-0 flex-col gap-4">
        <div>
          <h3 className="nx-eyebrow">Quality checks the pipeline ran</h3>
          <p className="mt-1 text-[13.5px] text-n-ink-2" data-testid="lineage-qc-summary">
            {L.checks.length ? (L.failed ? `${L.failed} of ${L.checks.length} checks failed (listed first).` : `All ${L.checks.length} checks passed.`) : "No checks are published for this layer."}
            {L.coverage ? ` ${L.coverage}.` : ""}
          </p>
          <ul className="mt-2 flex flex-col gap-1 text-[13px]" data-testid="lineage-qc">
            {checks.map((q, i) => (
              <li key={`${q.name}-${i}`} className="flex gap-2" data-passed={q.passed}>
                <span className={`mt-px shrink-0 ${q.passed ? "text-n-ok" : "text-n-down"}`}>
                  <Icon name={q.passed ? "check" : "alert"} className="h-4 w-4" />
                  <span className="sr-only">{q.passed ? "passed" : "failed"}</span>
                </span>
                <span className="min-w-0 break-words">
                  <b className="font-semibold">{q.name}</b>
                  {q.passed ? "" : " · failed"} · <span className="text-n-ink-2">{q.detail}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="nx-eyebrow">Uncertainty and caveats</h3>
          {L.quantErr != null && (
            <p className="mt-1 text-[13.5px] leading-snug" data-testid="lineage-quant">
              {L.quantLog ? (
                <>
                  Values are stored as 16-bit integers of log10(value): each decoded value is within <span className="nx-data">± {Number(L.quantErr.toPrecision(2))}</span> in log10, about ± {Number(((10 ** L.quantErr - 1) * 100).toPrecision(2))}% of the producer&rsquo;s value.
                </>
              ) : (
                <>
                  Values are stored as 16-bit integers: each decoded value is within <span className="nx-data">± {Number(L.quantErr.toPrecision(2))}</span> of the producer&rsquo;s.
                </>
              )}{" "}
              This layer carries no uncertainty interval for the values themselves.
            </p>
          )}
          <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-[13px] leading-snug text-n-ink-2" data-testid="lineage-caveats">
            {L.caveats.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
