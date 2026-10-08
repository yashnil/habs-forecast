import type { Manifest } from "@/generated/schema";
import { isFixture } from "@/lib/layers";
import { formatDateTimePT } from "@/lib/time";

export function DataBanners({ manifest, error }: { manifest: Manifest | null; error: string | null }) {
  if (error || !manifest) {
    return (
      <div role="alert" data-testid="data-unavailable" className="border-b border-serious/40 bg-serious/10 px-4 py-2 text-[12.5px] text-ink">
        <strong className="font-semibold">Live data unavailable.</strong> The forecast and satellite layers cannot be shown right now
        {error ? ` (${error})` : ""}. Official closures and advisories are always available from CDFW and CDPH.
      </div>
    );
  }
  const failed = manifest.sources.filter((s) => s.outcome === "failed");
  return (
    <>
      {isFixture(manifest) && (
        <div role="status" data-testid="fixture-banner" className="border-b border-warning/40 bg-warning/10 px-4 py-2 text-[12.5px] text-ink">
          <strong className="font-semibold">Test data.</strong> This build shows recorded fixture data (C-HARM subset for Monterey Bay and the
          Gulf of the Farallones, recorded {formatDateTimePT("2026-10-08T17:02:00Z")}), not a live feed.
        </div>
      )}
      {failed.length > 0 && (
        <div role="status" data-testid="source-failure-banner" className="border-b border-warning/40 bg-warning/10 px-4 py-2 text-[12.5px] text-ink">
          Latest update failed for {failed.map((s) => s.title).join(", ")}. The last successful data is shown with its real dates.{" "}
          <a href="/sources" className="font-medium underline">
            Details
          </a>
        </div>
      )}
    </>
  );
}
