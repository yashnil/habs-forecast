import type { ObsStation, ObservationDataset } from "@/generated/observations";
import { BLOOM_COPY } from "@/content/copy";
import { fmtDate } from "@/components/charts/StationChart";
import { SourceLink } from "@/components/ui/Primitives";
import { formatObs, sourceCsvUrl, summaryOf, unitLabel } from "@/lib/observations";
import { formatDateTimePT } from "@/lib/time";

const NOTES: { title: string; body: string }[] = [
  {
    title: "What a value is",
    body: "One laboratory result from one water sample taken at this pier on the date shown. It does not describe nearby beaches, offshore water or fishing grounds. Results arrive days to weeks after sampling.",
  },
  {
    title: "Blanks, zeros and uncertainty",
    body: "A blank means the quantity was not measured that day; it is not a zero. A reported 0 means not quantified, at or below a limit the laboratories do not publish. They publish no measurement uncertainty either, so none is shown.",
  },
  {
    title: "Seawater is not seafood",
    body: `${BLOOM_COPY.notSeafood} The model band is a probability for nearby ocean cells, a different quantity, and is never compared with the measurements.`,
  },
];

function Note({ title, compact, children }: { title: string; compact: boolean; children: string }) {
  if (compact)
    return (
      <details className="py-2.5">
        <summary className="cursor-pointer font-semibold text-ink">{title}</summary>
        <p className="mt-1">{children}</p>
      </details>
    );
  return (
    <div>
      <h3 className="mb-1 text-[14px] font-semibold text-ink">{title}</h3>
      <p>{children}</p>
    </div>
  );
}

type Dataset = Pick<ObservationDataset, "variables" | "method" | "caveats" | "provenance" | "generated_at" | "program">;

/**
 * What the numbers mean (spec §3.2 item 10), in plain language first: three short columns, then
 * disclosures for the details. Uncertainty is stated, not invented: the laboratories publish no
 * measurement uncertainty or detection limit.
 */
export function AboutData({ station, dataset, compact = false }: { station: ObsStation; dataset: Dataset; compact?: boolean }) {
  return (
    <section aria-labelledby="about-h" className="space-y-4" data-testid="about-data">
      <h2 id="about-h" className="font-display text-[24px] font-medium leading-tight text-ink">
        About these data
      </h2>
      <div className={compact ? "divide-y divide-hairline border-y border-hairline text-[14.5px] leading-relaxed text-ink-2" : "grid gap-x-8 gap-y-4 text-[14.5px] leading-relaxed text-ink-2 md:grid-cols-3"} data-testid="bloom-caveats">
        {NOTES.map((n) => (
          <Note key={n.title} title={n.title} compact={compact}>
            {n.body}
          </Note>
        ))}
      </div>

      <div className="divide-y divide-hairline border-y border-hairline text-[13.5px]">
        <details className="group py-2.5">
          <summary className="cursor-pointer font-medium text-ink">What each quantity is, and its latest value here</summary>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-[13px]" data-testid="variable-table">
              <thead className="text-ink-3">
                <tr>
                  <th className="py-1 pr-3 font-medium">Quantity</th>
                  <th className="py-1 pr-3 font-medium">Latest here</th>
                  <th className="py-1 pr-3 font-medium">Matrix, fraction</th>
                  <th className="py-1 pr-3 font-medium">Method</th>
                  <th className="py-1 font-medium">Detection limit</th>
                </tr>
              </thead>
              <tbody>
                {dataset.variables.map((v) => {
                  const sm = summaryOf(station, v.id);
                  return (
                    <tr key={v.id} className="border-t border-hairline align-top">
                      <td className="py-1.5 pr-3 text-ink">
                        {v.label} <span className="text-ink-3">({unitLabel(v.units)})</span>
                        <span className="block font-mono text-[12px] text-ink-3">{v.source_variable}</span>
                      </td>
                      <td className="py-1.5 pr-3 tabular" data-testid={`latest-${v.id}`}>
                        {sm && sm.n_measured > 0 && sm.last_date ? (
                          <>
                            {formatObs(v, sm.last_value ?? null, sm.last_qualifier ?? null)}
                            <span className="block text-ink-3">{fmtDate(Date.parse(`${sm.last_date}T12:00:00Z`))}</span>
                          </>
                        ) : (
                          <span className="text-ink-3">never measured here</span>
                        )}
                      </td>
                      <td className="py-1.5 pr-3">
                        {v.matrix}
                        {v.fraction ? `, ${v.fraction}` : ""}
                      </td>
                      <td className="py-1.5 pr-3">{v.method}</td>
                      <td className="py-1.5">
                        {v.detection_limit ?? "not published"}
                        <span className="block text-ink-3">{v.zero_policy}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
        <details className="py-2.5">
          <summary className="cursor-pointer font-medium text-ink">Quality checks for this station</summary>
          <ul className="mt-2 space-y-1" data-testid="station-qc">
            {station.qc.map((q) => (
              <li key={q.name}>
                <span className={q.passed ? "text-ink-3" : "text-warning"}>{q.passed ? "✓" : "⚠"}</span> <span className="font-medium text-ink">{q.name}</span>: {q.detail}
              </li>
            ))}
          </ul>
        </details>
        <details className="py-2.5">
          <summary className="cursor-pointer font-medium text-ink">How CoastWatch processes these data</summary>
          <dl className="mt-2 grid gap-x-4 gap-y-1.5 sm:grid-cols-[auto_1fr]">
            {Object.entries(dataset.method).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="capitalize text-ink-3">{k}</dt>
                <dd className="text-ink-2">{v}</dd>
              </div>
            ))}
          </dl>
        </details>
        <details className="py-2.5">
          <summary className="cursor-pointer font-medium text-ink">All caveats</summary>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-ink-2">
            {dataset.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </details>
      </div>

      <SourceCsv station={station} />

      <p className="text-[13px] leading-relaxed text-ink-3" data-testid="bloom-provenance">
        {station.lat != null && station.lon != null && (
          <>
            Station <span className="font-mono">{station.location_code ?? station.station_id}</span> at{" "}
            <span className="font-mono">
              {station.lat.toFixed(3)}°N {Math.abs(station.lon).toFixed(3)}°W
            </span>
            .{" "}
          </>
        )}
        Source: <SourceLink href={station.source_url}>{station.station_id} on SCCOOS ERDDAP ↗</SourceLink> · <SourceLink href={dataset.provenance.source_url}>CalHABMAP ↗</SourceLink>. Retrieved{" "}
        {formatDateTimePT(dataset.generated_at)}. C-HARM v3.1 model history: NOAA CoastWatch ERDDAP.
      </p>
      <details className="text-[13px] text-ink-3">
        <summary className="cursor-pointer">Citation and licence</summary>
        <p className="mt-1 leading-relaxed">
          {dataset.provenance.citation} {dataset.provenance.license}
        </p>
      </details>
      <p className="text-[13px] text-ink-3" data-testid="science-review-pending">
        {BLOOM_COPY.reviewPending}
      </p>
    </section>
  );
}

/**
 * The station's original data, straight from SCCOOS ERDDAP (the query the pipeline ran). Opens in a
 * new tab, so a source outage shows ERDDAP's error there and leaves this page as it was.
 */
function SourceCsv({ station }: { station: ObsStation }) {
  const href = sourceCsvUrl(station);
  return (
    <div className="space-y-1.5 text-[14px] leading-relaxed text-ink-2" data-testid="source-csv">
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex min-h-[40px] items-center gap-2 rounded-lg border border-hairline-strong bg-surface px-3.5 font-medium text-ink hover:border-accent hover:text-accent"
          data-testid="source-csv-link"
        >
          <span aria-hidden>↓</span> {BLOOM_COPY.csvLabel}
          <span className="sr-only"> for {station.name}, from SCCOOS ERDDAP (opens in a new tab)</span>
        </a>
      ) : null}
      <p className="max-w-[72ch] text-[13px] text-ink-3">
        {href ? `${BLOOM_COPY.csvWhat} ` : null}
        {station.status !== "updated" ? `${BLOOM_COPY.csvStale} ` : null}
        {href ? BLOOM_COPY.csvOutage : "The source data for this station are on its dataset page."}{" "}
        <SourceLink href={station.source_url}>Dataset page ↗</SourceLink>
      </p>
    </div>
  );
}
