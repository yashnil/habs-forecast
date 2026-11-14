"use client";

import { ErrorBody } from "@/nx/pages/errors";

/** A page of the group that failed to draw: the shell and the chart stay; the page says so (src/nx/pages/errors.tsx). */
export default function GroupError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return <ErrorBody retry={reset} />;
}
