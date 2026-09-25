"use client";

/**
 * Compliance-first hierarchy: official sources and rules override every model layer.
 * Collapsed by default so the map stays the focus; expand when you need links.
 */
export default function ComplianceBanner() {
  return (
    <details className="group border-b border-amber-900/45 bg-gradient-to-r from-amber-950/85 via-slate-950 to-slate-950 text-slate-200">
      <summary className="mx-auto flex max-w-6xl cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 marker:hidden [&::-webkit-details-marker]:hidden">
        <span className="text-xs font-semibold tracking-wide text-amber-200/95">
          Official rules always win (CDPH · CDFW · OEHHA · MPAs)
        </span>
        <span className="shrink-0 text-[11px] text-slate-500 group-open:hidden">
          Expand for details and links
        </span>
        <span className="hidden shrink-0 text-[11px] text-slate-500 group-open:inline">
          Collapse
        </span>
      </summary>
      <div className="mx-auto max-w-6xl border-t border-slate-800/80 px-4 pb-4 pt-3">
        <ol className="list-decimal space-y-1.5 pl-4 text-xs leading-relaxed text-slate-300">
          <li>
            <strong className="text-slate-100">Closures, MPAs, seasons, and health advisories</strong>{" "}
            come from California and federal agencies. This app does <strong>not</strong> replace
            them. If an area is closed or under advisory, <strong>do not</strong> treat green or
            low chlorophyll on the map as permission to harvest.
          </li>
          <li>
            <strong className="text-slate-100">Satellite chlorophyll</strong> (NASA GIBS) is
            open browse imagery over the ocean — context only, not toxin or legal status.
          </li>
          <li>
            <strong className="text-slate-100">Apps and informal forecasts</strong> are not
            regulatory sources — always confirm closures and health notices with the agencies
            above.
          </li>
        </ol>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 border-t border-slate-800/80 pt-3 text-[11px]">
          <span className="font-medium text-slate-500">Official (new tab):</span>
          <a
            className="text-cyan-400 hover:underline"
            href="https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/MarineBiotech.aspx"
            target="_blank"
            rel="noreferrer"
          >
            CDPH — marine biotoxins
          </a>
          <a
            className="text-cyan-400 hover:underline"
            href="https://oehha.ca.gov/fish/general-info/domoic-acid"
            target="_blank"
            rel="noreferrer"
          >
            OEHHA — domoic acid
          </a>
          <a
            className="text-cyan-400 hover:underline"
            href="https://wildlife.ca.gov/Fishing/Ocean"
            target="_blank"
            rel="noreferrer"
          >
            CDFW — ocean fishing
          </a>
          <a
            className="text-cyan-400 hover:underline"
            href="https://wildlife.ca.gov/Fishing/Ocean/Regulations"
            target="_blank"
            rel="noreferrer"
          >
            CDFW — regulations
          </a>
          <a
            className="text-cyan-400 hover:underline"
            href="https://www.mpamap.org/"
            target="_blank"
            rel="noreferrer"
          >
            California MPA Map
          </a>
          <a
            className="text-cyan-400 hover:underline"
            href="https://www.fisheries.noaa.gov/west-coast/ecosystems/harmful-algal-blooms-west-coast"
            target="_blank"
            rel="noreferrer"
          >
            NOAA WC — harmful algae
          </a>
        </div>
      </div>
    </details>
  );
}
