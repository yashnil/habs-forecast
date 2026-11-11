"use client";

import { Unavailable } from "@/nx/ds/States";
import { ChartScene } from "@/nx/chart/scene";
import { CA_COAST } from "@/nx/shell/views";
import { plainError } from "@/nx/ds/format";

/** Research when the dataset cannot be read: what is missing, why, and what to do (README §9 states). */
export function ResearchUnavailable({ why }: { why: string }) {
  return (
    <section data-testid="research-unavailable">
      <ChartScene scene={{ id: "research:unavailable", view: { bounds: CA_COAST } }} />
      <p className="n12-kicker">Research</p>
      <Unavailable what="The station records can't be read right now" why={`Neritic could not read its published dataset (${plainError(why)}). Nothing is shown rather than an old or partial record.`} next={<a href="/sources">See the data status</a>} />
    </section>
  );
}

