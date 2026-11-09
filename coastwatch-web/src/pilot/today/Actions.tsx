"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { Readiness } from "@/pilot/contracts";
import { READINESS_WORDS } from "@/pilot/readiness";
import { Icon } from "@/v4/ui/Icon";
import { readSaved } from "./prefs";
import { saveTodayOffline } from "./offline";
import { Section } from "./ui";

const PT = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });

type SaveState = { phase: "idle" } | { phase: "saving" } | { phase: "saved"; at: string } | { phase: "failed"; why: string };

/** The page's actions: save offline (the field-pack mechanism), alerts and Ask (other tracks), and the supporting views. */
export function Actions({ place, label, conditionsUrl, readiness }: { place: string; label: string; conditionsUrl: string; readiness: { alerts: Readiness; ask: Readiness } }) {
  const [save, setSave] = useState<SaveState>({ phase: "idle" });
  useEffect(() => {
    const s = readSaved().find((x) => x.place === place);
    if (s) setSave({ phase: "saved", at: s.at });
  }, [place]);
  const port = place.replace(/^port-/, "");
  const q = `place=${encodeURIComponent(place)}`;
  return (
    <Section id="actions" area="actions" title="Before you go">
      <div className="mt-4 grid gap-2 sm:grid-cols-3">
        <button
          type="button"
          className="today-btn primary nx-focus nx-press w-full"
          disabled={save.phase === "saving"}
          onClick={async () => {
            setSave({ phase: "saving" });
            const r = await saveTodayOffline(place, label, conditionsUrl);
            setSave(r.ok ? { phase: "saved", at: r.at } : { phase: "failed", why: r.why });
          }}
          data-testid="save-offline"
        >
          <Icon name={save.phase === "saved" ? "check" : "download"} />
          {save.phase === "saving" ? "Saving…" : save.phase === "saved" ? "Saved for offline · update" : "Save for offline"}
        </button>
        <Link href={`/account?${q}`} className="today-action nx-focus nx-press nx-tint" data-testid="action-alert">
          <span className="flex items-center gap-2 text-[15.5px] font-semibold text-n-ink">
            <Icon name="bell" className="h-[18px] w-[18px]" /> Set an alert
          </span>
          <small data-testid="readiness-alerts">{READINESS_WORDS[readiness.alerts]}: a text or push when this port&rsquo;s status changes</small>
        </Link>
        <Link href={`/ask?${q}`} className="today-action nx-focus nx-press nx-tint" data-testid="action-ask">
          <span className="flex items-center gap-2 text-[15.5px] font-semibold text-n-ink">
            <Icon name="ask" className="h-[18px] w-[18px]" /> Ask Neritic
          </span>
          <small data-testid="readiness-ask">{READINESS_WORDS[readiness.ask]}: answers from these facts, with sources</small>
        </Link>
      </div>
      <p className="mt-2 text-[13.5px] text-n-ink-3" role="status" aria-live="polite" data-testid="save-status">
        {save.phase === "saved" && `Saved ${PT.format(new Date(save.at))}. This page opens without a connection, as it was then.`}
        {save.phase === "failed" && `Could not save for offline: ${save.why}`}
        {save.phase === "idle" && "Keeps this page, its sea and weather, and its code on this device for use without signal."}
      </p>
      <nav aria-label="More about this port" className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-[15px]">
        <Link href={`/?sel=harbor:${port}`} className="nx-focus inline-flex min-h-11 items-center gap-1.5 text-n-ink underline decoration-n-line-strong underline-offset-4" data-testid="link-explore">
          <Icon name="map" className="h-4 w-4" /> Explore the map
        </Link>
        <Link href={`/coast?${q}`} className="nx-focus inline-flex min-h-11 items-center gap-1.5 text-n-ink underline decoration-n-line-strong underline-offset-4">
          <Icon name="harbor" className="h-4 w-4" /> Trip briefing
        </Link>
        <Link href={`/research?${q}`} className="nx-focus inline-flex min-h-11 items-center gap-1.5 text-n-ink underline decoration-n-line-strong underline-offset-4">
          <Icon name="research" className="h-4 w-4" /> Research
        </Link>
        <Link href="/sources" className="nx-focus inline-flex min-h-11 items-center gap-1.5 text-n-ink underline decoration-n-line-strong underline-offset-4">
          <Icon name="book" className="h-4 w-4" /> Sources and data status
        </Link>
      </nav>
    </Section>
  );
}
