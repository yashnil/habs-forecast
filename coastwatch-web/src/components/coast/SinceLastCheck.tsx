"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Mark, type MarkKind } from "@/components/ui/Mark";
import type { Snapshot } from "@/lib/briefingSnapshot";
import { diffSnapshots, type Change, type ChangeGroup } from "@/lib/changes";
import { coastHref, parseAudience } from "@/lib/coast";
import { defaultAudience, usePurpose } from "@/lib/purpose";
import { activityLabel, keptFilename, keptHtml, keptTitle, makeKept, saveKept, type Kept } from "@/lib/keptBriefing";
import { cacheKeptPage } from "@/v4/offline/kept";
import { readHome } from "@/v4/coast/home";
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
  // the kept copy opens offline too once its page is stored (null: not yet / could not be stored)
  const [offline, setOffline] = useState<boolean | null>(null);
  const [auto, setAuto] = useState(false);
  const [forgotten, setForgotten] = useState(false);

  useEffect(() => {
    if (!place || !purposeRead || !storageAvailable()) return;
    let live = true;
    setState({ phase: "idle" });
    setKept(null);
    setOffline(null);
    setAuto(false);
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
        // the home port's own trip is kept automatically, one copy replaced on each visit (bounded: the
        // kept list holds five), so it can be read offline without remembering to keep it
        const home = readHome();
        if (home && home.place === snap.place.key && home.for === snap.audience && home.species === snap.species.id) {
          const k = { ...makeKept(snap, new Date()), id: `home-${snap.place.key}` };
          if (saveKept(k)) {
            setKept(k);
            setAuto(true);
            void cacheKeptPage({ id: k.id, keptAt: k.keptAt, title: keptTitle(k) }).then((ok) => live && setOffline(ok));
          }
        }
      })
      .catch(() => live && setState({ phase: "error" }));
    return () => {
      live = false;
    };
  }, [place, audience, species, purposeRead]);

  if (!place || forgotten) return null;
  if (state.phase === "error")
    return (
      <p className="mt-3 text-[14px] text-n-ink-3" data-testid="since-error">
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
    if (!saveKept(k)) return;
    setKept(k);
    setAuto(false);
    setOffline(null);
    void cacheKeptPage({ id: k.id, keptAt: k.keptAt, title: keptTitle(k) }).then(setOffline);
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
  const checkedLine = snap.official.ok && snap.official.checkedAt ? `Neritic last checked the agencies' pages ${formatDateTimePT(snap.official.checkedAt)}.` : "Official notices could not be loaded.";
  const forget = () => {
    forgetAll();
    setForgotten(true);
  };

  // something changed: the list. Otherwise (a first visit, or nothing changed) one quiet grey line with
  // the two actions in it, never a card that pushes the briefing down (v6)
  const quiet = !changes.length;
  const actions = (
    <>
      {kept ? (
        <span data-testid="kept-done">
          <span className="mr-3 text-n-ink-2">
            {auto ? "Your home port, kept automatically" : "Kept"} {formatDateTimePT(kept.keptAt)}
            {offline ? <span data-testid="kept-offline">, opens offline</span> : offline === false ? ", on this device (opens offline only from a saved file)" : ""}.
          </span>
          <button type="button" onClick={() => download(kept)} className="nx-press nx-focus -my-3 mr-3 inline-flex min-h-[44px] items-center font-semibold text-n-accent" data-testid="kept-download">
            Save file
          </button>
          <Link href={`/coast/kept?id=${encodeURIComponent(kept.id)}`} className="-my-3 mr-3 inline-flex min-h-[44px] items-center font-semibold text-n-accent" data-testid="kept-open">
            Open
          </Link>
        </span>
      ) : (
        <button type="button" onClick={keep} className="nx-press nx-focus -my-3 mr-3 inline-flex min-h-[44px] items-center font-semibold text-n-accent" data-testid="kept-make">
          Keep a dated copy
        </button>
      )}
      <button type="button" onClick={forget} className="nx-press nx-focus -my-3 inline-flex min-h-[44px] items-center text-n-ink-3 underline decoration-n-line-strong underline-offset-2" data-testid="since-forget">
        Forget on this device
      </button>
    </>
  );

  return (
    <section
      aria-labelledby="since-h"
      className="max-w-[68ch]"
      data-testid="since-last-check" data-state={since ? (changes.length ? "changes" : "none") : "first"}>
      {resume ? (
        <p className="mb-2">
          <Link href={coastHref({ place: snap.place.key, for: parseAudience(resume.audience), species: resume.species })} className="nx-press nx-focus inline-flex min-h-[44px] items-center rounded-xl border border-n-line-strong px-4 text-[15px] font-semibold text-n-ink" data-testid="since-resume">
            Continue with {activityLabel(resume.audience)} · {resume.speciesLabel}
          </Link>
        </p>
      ) : null}
      {!quiet ? (
        <>
          <h2 id="since-h" className="text-[14px] font-semibold text-n-ink">
            Since you last checked <span className="font-normal text-n-ink-2">· {formatDateTimePT(since!)}</span>
          </h2>
          <ul className="mt-1 divide-y divide-n-line" data-testid="since-list">
            {shown.map((c, i) => (
              <li key={`${c.kind}-${i}`} className="grid grid-cols-[16px_1fr] gap-x-2 py-2" data-testid={`since-item-${c.kind}`} data-for-you={c.forYou}>
                <Mark kind={MARK[c.group]} aged={c.group === "coastwatch"} className="translate-y-[3px]" />
                <div className="min-w-0">
                  <p className="text-[15px] font-semibold leading-snug text-n-ink">{c.forYou ? "" : "Other species here: "}{c.title}</p>
                  <p className="text-[14px] leading-snug text-n-ink-2">{c.detail}</p>
                  {c.source ? (
                    <a href={c.source.url} target="_blank" rel="noopener noreferrer" className="inline-flex min-h-[44px] items-center text-[14px] font-semibold text-n-accent underline-offset-2 hover:underline">
                      {c.source.label} ↗
                    </a>
                  ) : null}
                </div>
              </li>
            ))}
          </ul>
          {hidden > 0 ? (
            <button type="button" onClick={() => setAll(true)} className="nx-press nx-focus min-h-[44px] text-[14px] font-semibold text-n-accent" data-testid="since-more">
              {hidden} more (other species, data and model)
            </button>
          ) : null}
          <p className="mt-2 text-[13px] text-n-ink-3" data-testid="since-checked">
            {checkedLine}
          </p>
          <div className="flex flex-wrap items-center gap-x-4 text-[14px]">{actions}</div>
        </>
      ) : since ? (
        // v7: a quiet state is one line (the caveat and the check time are for screen readers and the title)
        <p className="flex flex-wrap items-center gap-x-3 text-[13px] leading-6 text-n-ink-3" data-testid="since-none" title={checkedLine}>
          <span>
            <span id="since-h" className="font-semibold text-n-ink-2">No change in Neritic&rsquo;s information</span> since {formatDateTimePT(since)}
            <span className="sr-only">; that does not mean nothing changed at the agencies. </span>
            <span className="sr-only" data-testid="since-checked">{checkedLine}</span>
          </span>
          {actions}
        </p>
      ) : (
        <p className="flex flex-wrap items-center gap-x-3 text-[13px] leading-6 text-n-ink-3" data-testid="since-first" title={checkedLine}>
          <span>
            <span id="since-h" className="font-semibold text-n-ink-2">Remembered on this device</span>
            <span className="sr-only">, to list what changed next time. </span>
            <span className="sr-only" data-testid="since-checked">{checkedLine}</span>
          </span>
          {actions}
        </p>
      )}
      <p className="sr-only">Activity: {activityLabel(snap.audience)}. Species: {snap.species.label}. Last checked dates are shown in Pacific time.</p>
    </section>
  );
}
