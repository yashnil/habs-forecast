import type { Readiness } from "@/pilot/contracts";
import { READINESS_WORDS } from "@/pilot/readiness";

/** The honest readiness label shown next to every pilot feature that is not live. */
export function ReadinessBadge({ value, testId }: { value: Readiness; testId?: string }) {
  const tone = value === "live" ? "border-n-official-line text-n-ink-2" : value === "demo" ? "border-n-model-line text-n-model" : "border-n-line-strong text-n-ink-2";
  return (
    <span className={`inline-flex min-h-6 items-center rounded-full border px-2.5 text-[12.5px] font-semibold ${tone}`} data-testid={testId} data-readiness={value}>
      {READINESS_WORDS[value]}
    </span>
  );
}
