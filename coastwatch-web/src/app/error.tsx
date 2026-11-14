"use client";

import { ErrorBody } from "@/nx/pages/errors";
import { StandaloneShell } from "@/nx/pages/StandaloneShell";

/** A page outside the group that failed to draw (the catalog), in the shell. */
export default function RootError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <StandaloneShell>
      <ErrorBody retry={reset} />
    </StandaloneShell>
  );
}
