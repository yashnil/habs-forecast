"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Mark, type MarkKind } from "@/components/ui/Mark";
import type { Snapshot } from "@/lib/briefingSnapshot";
import { diffSnapshots, type Change, type ChangeGroup } from "@/lib/changes";
import { coastHref, parseAudience } from "@/lib/coast";
import { defaultAudience, usePurpose } from "@/lib/purpose";
import { activityLabel, keptFilename, keptHtml, makeKept, saveKept, type Kept } from "@/lib/keptBriefing";
import { forgetAll, keepSessionBaseline, recordVisit, rememberedFor, sessionBaseline, storageAvailable, watersKey, type SavedWaters } from "@/lib/savedWaters";
import { formatDate, formatDateTimePT } from "@/lib/time";

/**
 * "Since you last checked" for the waters, activity and species in the page's link. Mounted once,
 * below My Coast's answer line, with no props (docs/coastwatch/fishermen-first/06-integration-contract.md).
 *
 * It remembers the waters on this device only. On a return visit it lists what differs from the
 * last visit, each item with its source and the date that applies. It also offers a dated copy to
 * keep or print. With storage blocked it renders nothing, so it never claims "nothing changed".
 */
export function SinceLastCheck() {
  return (
    <Suspense fallback={null}>
      <SinceLastCheckInner />
    </Suspense>
  );
}

type State =
  | { phase: "idle" }
  | { phase: "ready"; snap: Snapshot; previous: SavedWaters | null; since: string | null; changes: Change[]; resume: SavedWaters | null }
  | { phase: "error" };

const MARK: Record<ChangeGroup, MarkKind> = { agency: "official", coastwatch: "official", observation: "measured", forecast: "model", data: "gap" };
const SHOWN = 4;
const fmt = { fmtDate: (d: string) => formatDate(d), fmtTime: (iso: string) => formatDateTimePT(iso) };

function SinceLastCheckInner() {
  const q = useSearchParams();
  const place = q.get("place");
  // the same rule as My Coast: a link's own filter wins, otherwise the saved purpose's default
  const [purpose, , purposeRead] = usePurpose();
  const audience = q.get("for") ? parseAudience(q.get("for")) : defaultAudience(purpose);
  const species = q.get("species") ?? "any";
  const [state, setState] = useState<State>({ phase: "idle" });
  const [all, setAll] = useState(false);
  const [kept, setKept] = useState<Kept | null>(null);
  const [forgotten, setForgotten] = useState(false);

  useEffect(() => {
    if (!place || !purposeRead || !storageAvailable()) return;
    let live = true;
    setState({ phase: "idle" });
    setKept(null);
    fetch(`/api/briefing?${new URLSearchParams({ place, for: audience, species })}`)
      .then((r) => (r.ok ? (r.json() as Promise<Snapshot>) : Promise.reject(new Error(String(r.status)))))
      .then((snap) => {
        if (!live) return;
        const key = watersKey(snap.place.key, snap.audience, snap.species.id);
        const resume = audience === "all" && species === "any" ? rememberedFor(snap.place.key) : null;
        let base = sessionBaseline(key);
        const { previous } = recordVisit(snap, new Date());
        if (!base && previous) {
          base = { key, since: previous.lastSeenAt, baseline: previous.snapshot };
          keepSessionBaseline(base);
        }
        const changes = base ? diffSnapshots(base.baseline, snap, base.since, fmt) : [];
        setState({ phase: "ready", snap, previous, since: base?.since ?? null, changes, resume });
      })
      .catch(() => live && setState({ phase: "error" }));
    return () => {
      live = false;
    };
  }, [place, audience, species, purposeRead]);

  if (!place || forgotten) return null;
  if (state.phase === "error")
    return (
      <p className="mt-3 text-[14px] text-ink-3" data-testid="since-error">
        Could not compare with your last visit just now.
      </p>
    );
  if (state.phase !== "ready") return null;

  const { snap, since, changes, resume } = state;
  const mine = changes.filter((c) => c.forYou);
  const shown = all ? changes : mine.slice(0, SHOWN);
  const hidden = changes.length - shown.length;
  const keep = () => {
    const k = makeKept(snap, new Date());
    if (saveKept(k)) setKept(k);
  };
  const download = (k: Kept) => {
    const url = URL.createObjectURL(new Blob([keptHtml(k)], { type: "text/html" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = keptFilename(k);
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const checkedLine = snap.official.ok && snap.official.checkedAt ? `Neritic Signal last checked the agencies' pages ${formatDateTimePT(snap.official.checkedAt)}.` : "Official notices could not be loaded.";
  const forget = () => {
    forgetAll();
    setForgotten(true);
  };

  return (
    <section
      aria-labelledby="since-h"
      // a first visit has nothing to report: one quiet line, not a card that pushes the briefing down
      className={since || resume ? "mt-4 max-w-[62ch] rounded-[14px] border border-hairline bg-surface px-4 py-3" : "mt-3 max-w-[62ch]"}
      data-testid="since-last-check" data-state={since ? (changes.length ? "changes" : "none") : "first"}>
      {resume ? (
        <p className="mb-2">
          <Link href={coastHref({ place: snap.place.key, for: parseAudience(resume.audience), species: resume.species })} className="cw-press inline-flex min-h-[44px] items-center rounded-full border border-hairline px-4 text-[15px] font-medium text-accent" data-testid="since-resume">
            Continue with {activityLabel(resume.audience)} · {resume.speciesLabel}
          </Link>
        </p>
      ) : null}
      {since ? (
        <>
          <h2 id="since-h" className="text-[15px] font-semibold text-ink">
            Since you last checked <span className="font-normal text-ink-2">· {formatDateTimePT(since)}</span>
          </h2>
          {changes.length ? (
            <ul className="mt-1 divide-y divide-hairline" data-testid="since-list">
              {shown.map((c, i) => (
                <li key={`${c.kind}-${i}`} className="grid grid-cols-[16px_1fr] gap-x-2 py-2" data-testid={`since-item-${c.kind}`} data-for-you={c.forYou}>
                  <Mark kind={MARK[c.group]} aged={c.group === "coastwatch"} className="translate-y-[3px]" />
                  <div className="min-w-0">
                    <p className="text-[15px] font-medium leading-snug text-ink">{c.forYou ? "" : "Other species here: "}{c.title}</p>
                    <p className="text-[14px] leading-snug text-ink-2">{c.detail}</p>
                    {c.source ? (
                      <a href={c.source.url} target="_blank" rel="noopener noreferrer" className="inline-flex min-h-[44px] items-center text-[14px] font-medium text-accent underline-offset-2 hover:underline">
                        {c.source.label} ↗
                      </a>
                    ) : null}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1 text-[15px] text-ink-2" data-testid="since-none">
              No change in Neritic Signal&rsquo;s information for these waters since then. That does not mean nothing changed at the agencies.
            </p>
          )}
          {hidden > 0 ? (
            <button type="button" onClick={() => setAll(true)} className="cw-press min-h-[44px] text-[14px] font-medium text-accent" data-testid="since-more">
              {hidden} more (other species, data and model)
            </button>
          ) : null}
        </>
      ) : (
        <p className="text-[13px] leading-snug text-ink-2" data-testid="since-first">
          <span id="since-h" className="font-medium text-ink">Remembered on this device</span>, to list what changed next time.{" "}
          <span data-testid="since-checked">{checkedLine}</span>
        </p>
      )}
      {since ? (
        <p className="mt-2 text-[13px] text-ink-3" data-testid="since-checked">
          {checkedLine}
        </p>
      ) : null}
      <div className={`${since ? "mt-1" : ""} flex flex-wrap items-center gap-x-4`}>
        {kept ? (
          <span className="flex flex-wrap items-center gap-x-4 text-[14px]" data-testid="kept-done">
            <span className="text-ink-2">Kept {formatDateTimePT(kept.keptAt)}.</span>
            <button type="button" onClick={() => download(kept)} className="cw-press min-h-[44px] font-medium text-accent" data-testid="kept-download">
              Save file
            </button>
            <Link href={`/coast/kept?id=${encodeURIComponent(kept.id)}`} className="inline-flex min-h-[44px] items-center font-medium text-accent" data-testid="kept-open">
              Print
            </Link>
          </span>
        ) : (
          <button type="button" onClick={keep} className="cw-press min-h-[44px] text-[14px] font-medium text-accent" data-testid="kept-make">
            Keep a dated copy
          </button>
        )}
        <button type="button" onClick={forget} className="cw-press min-h-[44px] text-[14px] text-ink-3 underline decoration-ink-3/40 underline-offset-2" data-testid="since-forget">
          Forget on this device
        </button>
      </div>
      <p className="sr-only">Activity: {activityLabel(snap.audience)}. Species: {snap.species.label}. Last checked dates are shown in Pacific time.</p>
    </section>
  );
}
