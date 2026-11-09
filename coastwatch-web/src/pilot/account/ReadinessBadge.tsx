import type { Readiness } from "@/pilot/contracts";
import { READINESS_WORDS } from "@/pilot/readiness";

/** The honest readiness label shown next to every pilot feature that is not live: a caps word, never a coloured pill. */
export function ReadinessBadge({ value, testId }: { value: Readiness; testId?: string }) {
  return (
    <span className="pilot-badge" data-testid={testId} data-readiness={value}>
      {READINESS_WORDS[value]}
    </span>
  );
}
