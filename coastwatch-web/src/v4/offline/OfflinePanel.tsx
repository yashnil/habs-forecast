"use client";

import { toast } from "@/v4/ui/toast";
import { useEffect, useMemo, useState } from "react";
import type { Manifest } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import { charmRun, satelliteLatest, sourceStatus } from "@/lib/layers";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayYearLabel } from "@/v4/ui/time";
import { useSetSource } from "@/v4/readiness/store";
import { GROUP_WORDS, PACK_VERSION, deletePack, download, isQuota, keepPublishedTiles, pageAssetsOf, plan, sizeOf, type Group, type PackDef, type PackProblem, type PackRecord, type PlanItem, type Progress } from "./packs";
import { packAge, useOfflinePack } from "./useOfflinePack";
import { getRun, packChanged, setRun, useRun } from "./runStore";

const mb = (b: number) => `${(b / 1_048_576).toFixed(b < 10_485_760 ? 1 : 0)} MB`;

/**
 * Save a region for use without a connection. The reader sees what will be saved, how large it is,
 * what will and will not work offline, and starts the download; progress is real bytes. After the
 * download the pack is verified (every file, and the basemap's checksum).
 */
export function OfflinePanel({ pack, manifest, baseUrl, onClose }: { pack: PackDef; manifest: Manifest; official?: OfficialDataset | null; baseUrl: string; onClose?: () => void }) {
  const { record, problem, online, sw, refresh, evicted } = useOfflinePack(pack.id);
  const [items, setItems] = useState<PlanItem[] | null>(null);
  const [planErr, setPlanErr] = useState<string | null>(null);
  // the download itself lives in a module store: closing this panel neither stops nor forgets it
  const run = useRun(pack.id);
  const prog: Progress | null = run.prog;
  const [sizing, setSizing] = useState(false);
  const state = run.state === "downloading" ? "downloading" : sizing ? "sizing" : planErr ? "failed" : run.state;
  const [quota, setQuota] = useState<{ usage: number; quota: number } | null>(null);
  const [persisted, setPersisted] = useState<boolean | null>(null);
  // removing a pack takes two steps: a field pack is minutes of download, and gone with one tap
  const [confirmDelete, setConfirmDelete] = useState(false);
  const setSource = useSetSource();

  useEffect(() => {
    navigator.storage?.estimate?.().then((e) => e.quota && setQuota({ usage: e.usage ?? 0, quota: e.quota }));
    navigator.storage?.persisted?.().then(setPersisted);
  }, [record, state]);

  const prepare = async () => {
    setSizing(true);
    setPlanErr(null);
    try {
      const assets = await pageAssetsOf([`/?sel=harbor:${pack.port}&layer=forecast`, `/coast?place=${pack.place}`]);
      setItems(await sizeOf(await keepPublishedTiles(plan(pack, manifest, baseUrl, assets), manifest, baseUrl)));
    } catch (e) {
      setPlanErr((e as Error).message);
    }
    setSizing(false);
  };
  useEffect(() => {
    // reopened during a download: its progress is shown, nothing is measured again meanwhile
    if (online && !items && !sizing && getRun(pack.id).state !== "downloading") void prepare();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [online]);

  const byGroup = useMemo(() => {
    const g = new Map<Group, { n: number; bytes: number; unknown: number }>();
    for (const it of items ?? []) {
      const x = g.get(it.group) ?? { n: 0, bytes: 0, unknown: 0 };
      x.n++;
      if (it.bytes != null) x.bytes += it.bytes;
      else x.unknown++;
      g.set(it.group, x);
    }
    return g;
  }, [items]);
  const total = (items ?? []).reduce((a, x) => a + (x.bytes ?? 0), 0);

  const start = async () => {
    if (!items || getRun(pack.id).state === "downloading") return;
    const ac = new AbortController();
    setRun(pack.id, { state: "downloading", prog: null, ac });
    try {
      // ask the browser to keep it (it may still decide not to)
      if (navigator.storage?.persist) setPersisted(await navigator.storage.persist());
    } catch {
      // not available: the panel says storage may be evicted
    }
    setSource({ id: "pack", label: `Field pack: ${pack.name}`, state: "downloading", detail: "Saving the region for offline use" });
    try {
      const rec = await download(
        pack,
        items,
        {
          dataGeneratedAt: manifest.generated_at,
          charmIssued: charmRun(manifest)?.issued_date ?? null,
          satelliteNewest: satelliteLatest(manifest, "olci300")?.composite?.newest_observed_date ?? null,
          noticesCheckedAt: sourceStatus(manifest, "official")?.last_success_at ?? null,
        },
        (p) => setRun(pack.id, { prog: p }),
        ac.signal,
      );
      packChanged(pack.id);
      if (rec.complete && !ac.signal.aborted) toast(`${pack.name} saved for offline use`);
      setRun(pack.id, { state: ac.signal.aborted ? "cancelled" : rec.complete ? "done" : "failed", ac: null });
      setSource({ id: "pack", label: `Field pack: ${pack.name}`, state: rec.complete ? "ready" : "incomplete", detail: rec.complete ? `Saved ${mb(rec.totalBytes)}, verified` : `${rec.verified?.missing ?? "Some"} files missing; download again to finish`, asOf: rec.savedAt });
    } catch (e) {
      const q = isQuota(e);
      setRun(pack.id, { state: q ? "quota" : "failed", ac: null });
      setSource({ id: "pack", label: `Field pack: ${pack.name}`, state: "incomplete", detail: q ? "The device's storage for this site is full" : (e as Error).message });
      packChanged(pack.id);
    }
  };

  const age = record && record.complete ? packAge(record, new Date()) : null;
  // the bar becomes a check only when this view saw the download finish (reopened later: just the check)
  const [seen, setSeen] = useState(state);
  const [justSaved, setJustSaved] = useState(false);
  if (seen !== state) {
    setSeen(state);
    setJustSaved(seen === "downloading" && state === "done");
  }
  const pct = prog && prog.totalBytes ? Math.min(100, Math.round((prog.bytes / prog.totalBytes) * 100)) : null;
  const eta = prog?.rate && prog.totalBytes ? Math.max(0, (prog.totalBytes - prog.bytes) / prog.rate) : null;

  return (
    <div className="flex flex-col gap-4" data-testid="offline-panel">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="nx-eyebrow">Field pack · offline use</p>
          <h2 className="mt-1 font-serif text-[22px] font-semibold leading-tight">{pack.name}</h2>
          <p className="mt-1 text-[14px] text-n-ink-2">The map, the C-HARM run, the latest satellite view, currents and the {pack.name.split(" ")[0]} briefing, saved on this device for when there is no signal.</p>
        </div>
        {onClose && (
          <button type="button" aria-label="Close" onClick={onClose} className="nx-press nx-focus grid h-11 w-11 shrink-0 place-items-center rounded-lg text-n-ink-2 hover:bg-n-surface-2">
            <Icon name="close" />
          </button>
        )}
      </div>

      {record && (
        <div className={`rounded-xl border p-3 text-[14px] ${record.complete ? "border-n-measured-line bg-n-measured-bg" : "border-n-official-line bg-n-official-bg"}`} data-testid="pack-status" data-complete={record.complete}>
          <p className="font-semibold">{record.complete ? `Saved ${clockLabel(record.savedAt)} · ${mb(record.totalBytes)} · verified` : `Incomplete: ${record.verified?.missing ?? "?"} of ${record.items.length + (record.verified?.missing ?? 0)} files missing`}</p>
          <p className="mt-1 text-n-ink-2">
            Data published {clockLabel(record.dataGeneratedAt)}
            {record.charmIssued ? ` · C-HARM run of ${dayYearLabel(record.charmIssued)}` : ""}
            {record.noticesCheckedAt ? ` · notices last checked ${clockLabel(record.noticesCheckedAt)}` : ""}
            {record.osmDataAsOf ? ` · map data (OpenStreetMap) of ${dayYearLabel(record.osmDataAsOf)}` : ""}.
          </p>
          {age?.outdated && <p className="mt-1 font-semibold text-n-stale">More than a day old: download again when you have a connection.</p>}
          {record.complete && <HomeScreenHint />}
          {problem === "older-format" && (
            <p className="mt-1 font-semibold text-n-stale" data-testid="pack-older">
              Saved by an earlier version of Neritic (pack format {record.version}, now {PACK_VERSION}). What it holds still opens offline, each value with its own time; download again so every part does.
            </p>
          )}
          {problem === "missing-files" && (
            <p className="mt-1 font-semibold" data-testid="pack-missing">
              Some saved files are no longer on this device (the browser may have removed them). Finish the download to use {pack.name} offline.
            </p>
          )}
          {confirmDelete ? (
            <div className="mt-2 flex flex-wrap items-center gap-2" role="group" aria-label="Remove the field pack?">
              <p className="w-full font-semibold">Remove {pack.name} ({mb(record.totalBytes)}) from this device? Offline use needs a new download.</p>
              <button
                type="button"
                autoFocus
                onClick={async () => {
                  await deletePack(pack.id);
                  setConfirmDelete(false);
                  setRun(pack.id, { state: "idle", prog: null });
                  packChanged(pack.id);
                  await refresh();
                }}
                className="nx-press nx-focus min-h-11 rounded-lg bg-n-down px-3 font-semibold text-white"
                data-testid="pack-delete-confirm"
              >
                Remove
              </button>
              <button type="button" onClick={() => setConfirmDelete(false)} className="nx-press nx-focus min-h-11 rounded-lg border border-n-line-strong px-3 font-semibold" data-testid="pack-delete-keep">
                Keep it
              </button>
            </div>
          ) : (
            <button type="button" onClick={() => setConfirmDelete(true)} className="nx-press nx-focus mt-2 min-h-11 rounded-lg border border-n-line-strong px-3 font-semibold" data-testid="pack-delete">
              Remove from this device
            </button>
          )}
        </div>
      )}

      {!record && (problem === "unfinished" || problem === "damaged-record") && (
        <p className="rounded-xl border border-n-official-line bg-n-official-bg p-3 text-[14px] font-semibold" data-testid="pack-problem" data-problem={problem}>
          {problem === "unfinished"
            ? `An unfinished download of ${pack.name} is on this device: it is not complete and nothing in it is verified. Finish the download to use ${pack.name} offline.`
            : `The saved copy of ${pack.name} cannot be read (its record is damaged), so it is not counted as saved. Download again to replace it.`}
        </p>
      )}
      {!record && evicted && (
        <p className="rounded-xl border border-n-official-line bg-n-official-bg p-3 text-[14px] font-semibold" data-testid="pack-evicted">
          The copy saved {clockLabel(evicted)} is gone: the browser removed it (low storage or a site left unused). Download it again to use {pack.name} offline.
        </p>
      )}
      {!online && !record && <p className="text-[14.5px] font-semibold text-n-offline">You are offline. A field pack can only be saved with a connection.</p>}
      {sw === "unsupported" || sw === "failed" ? <p className="text-[14px] text-n-down">This browser cannot keep pages for offline use (no service worker).</p> : null}

      {online && (
        <>
          <section className="flex flex-col gap-2" data-testid="pack-plan">
            <h3 className="nx-eyebrow">What will be saved</h3>
            {state === "sizing" || (!items && !planErr) ? (
              <p className="text-[14px] text-n-ink-2">Measuring each part…</p>
            ) : planErr ? (
              <p className="text-[14px] text-n-down">The list could not be prepared: {planErr}.</p>
            ) : (
              <>
                <ul className="divide-y divide-n-line rounded-xl border border-n-line">
                  {[...byGroup.entries()].map(([g, x]) => (
                    <li key={g} className="flex items-start justify-between gap-3 px-3 py-2.5">
                      <span className="min-w-0">
                        <span className="block text-[14.5px] font-semibold">{GROUP_WORDS[g].name}</span>
                        <span className="block text-[13px] leading-snug text-n-ink-3">{GROUP_WORDS[g].offline}</span>
                      </span>
                      <span className="nx-data shrink-0 text-right text-[13.5px] text-n-ink-2">{x.bytes ? mb(x.bytes) : ""}{x.unknown ? <span className="block text-[12px] text-n-ink-3">{x.bytes ? "+ " : ""}{x.unknown} sized on saving</span> : null}</span>
                    </li>
                  ))}
                </ul>
                <p className="text-[14px]">
                  At least <span className="nx-data font-semibold">{mb(total)}</span> in {items?.length} files (pages and evidence are sized as they are saved)
                  {quota ? ` · ${mb(quota.quota - quota.usage)} free for this site` : ""}
                </p>
                {quota && total > quota.quota - quota.usage && (
                  <p className="text-[14px] font-semibold text-n-down" data-testid="pack-no-room">
                    Not enough free storage for this site: the download would stop part way. Free some space on the device first.
                  </p>
                )}
              </>
            )}
            <h3 className="nx-eyebrow mt-2">Not available offline</h3>
            <ul className="list-disc pl-5 text-[13.5px] leading-snug text-n-ink-2">
              <li>Anything new after the download: harbour water, buoys and NWS forecasts stay as saved, with their times.</li>
              <li>Official notices: Neritic&rsquo;s copy as last checked; the agencies&rsquo; own pages need a connection.</li>
              <li>Other harbours, earlier satellite days and the map outside {pack.name}: zoomed in beyond it, only the seafloor relief is drawn (no coastline, roads or names).</li>
              <li>Not for navigation.</li>
            </ul>
          </section>

          {state === "done" && <SavedMark play={justSaved} name={pack.name} />}
          {state === "downloading" && prog ? (
            <div className="flex flex-col gap-2" data-testid="pack-progress">
              <div className="h-2 w-full overflow-hidden rounded-full bg-n-surface-3" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct ?? undefined} aria-label="Download progress">
                <div className="h-full bg-n-accent transition-[width] duration-200" style={{ width: `${pct ?? 0}%` }} />
              </div>
              <p className="nx-data text-[13.5px] text-n-ink-2" aria-live="polite">
                {mb(prog.bytes)} of {mb(prog.totalBytes)} · {prog.done}/{prog.total} files{prog.current ? ` · ${GROUP_WORDS[prog.current].name.toLowerCase()}` : ""}
                {eta != null ? ` · about ${eta < 60 ? `${Math.max(1, Math.round(eta))} s` : `${Math.round(eta / 60)} min`} left` : ""}
              </p>
              <button type="button" onClick={() => getRun(pack.id).ac?.abort()} className="nx-press nx-focus min-h-12 rounded-xl border border-n-line-strong font-semibold" data-testid="pack-cancel">
                Stop
              </button>
            </div>
          ) : (
            <button
              type="button"
              disabled={!items || state === "sizing" || sw !== "ready"}
              onClick={start}
              // stays in view while the dialog scrolls (a phone sees the plan and the button together)
              className="nx-press nx-focus sticky bottom-0 z-10 flex min-h-14 items-center justify-center gap-2 rounded-xl bg-n-ink text-[16px] font-semibold text-n-bg shadow-[0_-8px_16px_var(--nx-surface)] disabled:opacity-40"
              data-testid="pack-download"
            >
              <Icon name="download" /> {downloadLabel(record, problem, manifest.generated_at, items ? mb(total) : "")}
            </button>
          )}
          {state === "cancelled" && <p className="text-[14px] text-n-ink-2" role="status">Stopped. What was saved is kept; download again to finish.</p>}
          {state === "quota" && (
            <p className="text-[14px] font-semibold text-n-down" role="alert" data-testid="pack-quota">
              The device&rsquo;s storage for this site is full: the pack is not complete{prog ? ` (${prog.done - prog.failed.length} of ${prog.total} files saved)` : ""}. What was saved is kept. Free some space, then finish the download.
            </p>
          )}
          {state === "failed" && prog && <p className="text-[14px] font-semibold text-n-down" role="alert">{prog.failed.length} files could not be saved. Download again to finish; nothing saved is lost.</p>}
          <p className="text-[12.5px] leading-snug text-n-ink-3">
            {persisted ? "The browser has agreed to keep this storage." : "The browser may remove saved data when the device is low on space or the site is unused; check the pack before you leave."} Map: © OpenStreetMap contributors (ODbL), © OpenMapTiles, OpenFreeMap. Relief: NOAA NCEI.
          </p>
        </>
      )}
    </div>
  );
}

/**
 * Saved and verified: the progress bar, full, gathers into a circle with a check, then the words come
 * (theme.css .nx-saved-*). The bar reached 100 % on real bytes first; this only marks the end.
 */
function SavedMark({ play, name }: { play: boolean; name: string }) {
  return (
    <div className={`relative ${play ? "nx-saved-play" : ""}`} data-testid="pack-saved">
      <span aria-hidden className="absolute inset-x-0 top-0 flex h-5 items-center">
        <span className="nx-saved-mark grid h-5 w-5 place-items-center overflow-hidden rounded-full bg-n-accent">
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0" fill="none" stroke="var(--nx-bg)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <path className="nx-saved-tick" pathLength={1} d="m5 12.5 4.5 4.5L19 7.5" />
          </svg>
        </span>
      </span>
      <p className="nx-saved-words pl-7 text-[14px] font-semibold leading-5 text-n-ok" role="status">
        Saved and verified. {name} now opens without a connection.
      </p>
    </div>
  );
}

/** Once a pack is saved: one quiet, dismissible hint, never shown in the installed app or after dismissal. */
const HINT_KEY = "nx-a2hs-hint-v1";
function HomeScreenHint() {
  const [show, setShow] = useState(false);
  useEffect(() => {
    try {
      const installed = matchMedia("(display-mode: standalone)").matches || (navigator as unknown as { standalone?: boolean }).standalone === true;
      setShow(!installed && localStorage.getItem(HINT_KEY) !== "1");
    } catch {
      setShow(false);
    }
  }, []);
  if (!show) return null;
  return (
    <p className="mt-2 flex items-center gap-2 text-[13.5px] text-n-ink-2" data-testid="a2hs-hint">
      <span className="min-w-0 flex-1">Add to Home Screen keeps saved data longer on iPhone.</span>
      <button
        type="button"
        onClick={() => {
          try {
            localStorage.setItem(HINT_KEY, "1");
          } catch {
            // not remembered: it is simply hidden for now
          }
          setShow(false);
        }}
        className="nx-press nx-focus min-h-11 shrink-0 rounded-lg px-2 font-semibold text-n-ink"
        data-testid="a2hs-dismiss"
      >
        Dismiss
      </button>
    </p>
  );
}

function downloadLabel(record: PackRecord | null | undefined, problem: PackProblem | null, generatedAt: string, size: string) {
  if (problem === "unfinished") return "Finish the download";
  if (!record) return `Download ${size}`;
  if (!record.complete) return "Finish the download";
  return record.dataGeneratedAt < generatedAt ? "Download the newer data" : "Download again";
}

export function useRecordSource(record: PackRecord | null | undefined, name: string, online: boolean, evicted: string | null = null, problem: PackProblem | null = null) {
  const setSource = useSetSource();
  useEffect(() => {
    if (record === undefined) return;
    if (!record) {
      if (problem === "unfinished" || problem === "damaged-record")
        setSource({ id: "pack", label: `Field pack: ${name}`, state: "incomplete", detail: problem === "unfinished" ? "Download not finished" : "The saved copy cannot be read; download it again" });
      else if (evicted) setSource({ id: "pack", label: `Field pack: ${name}`, state: "unavailable", detail: "Removed by the browser; download it again for offline use", asOf: evicted });
      return;
    }
    if (problem === "older-format" && record.complete) {
      setSource({ id: "pack", label: `Field pack: ${name}`, state: online ? "stale" : "cached", detail: online ? "Saved by an earlier version of Neritic; download again" : "Showing the saved copy (offline; saved by an earlier version of Neritic)", asOf: record.savedAt });
      return;
    }
    // more than a day old: C-HARM and the observations have moved on, so the copy is stale
    const outdated = record.complete && packAge(record, new Date()).outdated;
    setSource({
      id: "pack",
      label: `Field pack: ${name}`,
      state: !record.complete ? "incomplete" : !online ? "cached" : outdated ? "stale" : "ready",
      detail: !record.complete ? "Download incomplete" : !online ? "Showing the saved copy (offline)" : outdated ? `Saved more than a day ago (${mb(record.totalBytes)}); download again for newer data` : `Saved ${mb(record.totalBytes)}, verified`,
      asOf: record.savedAt,
    });
  }, [record, name, online, evicted, problem, setSource]);
}
