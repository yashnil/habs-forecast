"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Manifest } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import { charmRun, satelliteLatest, sourceStatus } from "@/lib/layers";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel } from "@/v4/ui/time";
import { useSetSource } from "@/v4/readiness/store";
import { GROUP_WORDS, deletePack, download, keepPublishedTiles, pageAssetsOf, plan, sizeOf, type Group, type PackDef, type PackRecord, type PlanItem, type Progress } from "./packs";
import { packAge, useOfflinePack } from "./useOfflinePack";

const mb = (b: number) => `${(b / 1_048_576).toFixed(b < 10_485_760 ? 1 : 0)} MB`;

/**
 * Save a region for use without a connection. The reader sees what will be saved, how large it is,
 * what will and will not work offline, and starts the download; progress is real bytes. After the
 * download the pack is verified (every file, and the basemap's checksum).
 */
export function OfflinePanel({ pack, manifest, baseUrl, onClose }: { pack: PackDef; manifest: Manifest; official?: OfficialDataset | null; baseUrl: string; onClose?: () => void }) {
  const { record, online, sw, refresh, evicted } = useOfflinePack(pack.id);
  const [items, setItems] = useState<PlanItem[] | null>(null);
  const [planErr, setPlanErr] = useState<string | null>(null);
  const [prog, setProg] = useState<Progress | null>(null);
  const [state, setState] = useState<"idle" | "sizing" | "downloading" | "done" | "failed" | "quota" | "cancelled">("idle");
  const [quota, setQuota] = useState<{ usage: number; quota: number } | null>(null);
  const [persisted, setPersisted] = useState<boolean | null>(null);
  const ac = useRef<AbortController | null>(null);
  const setSource = useSetSource();

  useEffect(() => {
    navigator.storage?.estimate?.().then((e) => e.quota && setQuota({ usage: e.usage ?? 0, quota: e.quota }));
    navigator.storage?.persisted?.().then(setPersisted);
  }, [record]);

  const prepare = async () => {
    setState("sizing");
    setPlanErr(null);
    try {
      const assets = await pageAssetsOf([`/?sel=harbor:${pack.port}&layer=forecast`, `/coast?place=${pack.place}`]);
      setItems(await sizeOf(await keepPublishedTiles(plan(pack, manifest, baseUrl, assets), manifest, baseUrl)));
      setState("idle");
    } catch (e) {
      setPlanErr((e as Error).message);
      setState("failed");
    }
  };
  useEffect(() => {
    if (online && !items && state === "idle") void prepare();
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
    if (!items) return;
    ac.current = new AbortController();
    setState("downloading");
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
        setProg,
        ac.current.signal,
      );
      await refresh();
      setState(ac.current.signal.aborted ? "cancelled" : rec.complete ? "done" : "failed");
      setSource({ id: "pack", label: `Field pack: ${pack.name}`, state: rec.complete ? "ready" : "incomplete", detail: rec.complete ? `Saved ${mb(rec.totalBytes)}, verified` : `${rec.verified?.missing ?? "Some"} files missing; download again to finish`, asOf: rec.savedAt });
    } catch (e) {
      const q = (e as Error).name === "QuotaExceededError";
      setState(q ? "quota" : "failed");
      setSource({ id: "pack", label: `Field pack: ${pack.name}`, state: "incomplete", detail: q ? "The device's storage for this site is full" : (e as Error).message });
      await refresh();
    }
  };

  const age = record && record.complete ? packAge(record, new Date()) : null;
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
            {record.charmIssued ? ` · C-HARM run of ${record.charmIssued}` : ""}
            {record.noticesCheckedAt ? ` · notices last checked ${clockLabel(record.noticesCheckedAt)}` : ""}
            {record.osmDataAsOf ? ` · map data (OpenStreetMap) of ${record.osmDataAsOf}` : ""}.
          </p>
          {age?.outdated && <p className="mt-1 font-semibold text-n-stale">More than a day old: download again when you have a connection.</p>}
          <button
            type="button"
            onClick={async () => {
              await deletePack(pack.id);
              await refresh();
              setState("idle");
              setProg(null);
            }}
            className="nx-press nx-focus mt-2 min-h-11 rounded-lg border border-n-line-strong px-3 font-semibold"
            data-testid="pack-delete"
          >
            Remove from this device
          </button>
        </div>
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
              </>
            )}
            <h3 className="nx-eyebrow mt-2">Not available offline</h3>
            <ul className="list-disc pl-5 text-[13.5px] leading-snug text-n-ink-2">
              <li>Anything new after the download: harbour water, buoys and NWS forecasts stay as saved, with their times.</li>
              <li>Official notices: Neritic&rsquo;s copy as last checked; the agencies&rsquo; own pages need a connection.</li>
              <li>Other harbours, earlier satellite days and the map outside {pack.name} (only the seafloor relief and coastline there).</li>
              <li>Not for navigation.</li>
            </ul>
          </section>

          {state === "downloading" && prog ? (
            <div className="flex flex-col gap-2" data-testid="pack-progress">
              <div className="h-2 w-full overflow-hidden rounded-full bg-n-surface-3" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct ?? undefined} aria-label="Download progress">
                <div className="h-full bg-n-accent transition-[width] duration-200" style={{ width: `${pct ?? 0}%` }} />
              </div>
              <p className="nx-data text-[13.5px] text-n-ink-2" aria-live="polite">
                {mb(prog.bytes)} of {mb(prog.totalBytes)} · {prog.done}/{prog.total} files{prog.current ? ` · ${GROUP_WORDS[prog.current].name.toLowerCase()}` : ""}
                {eta != null ? ` · about ${eta < 60 ? `${Math.max(1, Math.round(eta))} s` : `${Math.round(eta / 60)} min`} left` : ""}
              </p>
              <button type="button" onClick={() => ac.current?.abort()} className="nx-press nx-focus min-h-12 rounded-xl border border-n-line-strong font-semibold" data-testid="pack-cancel">
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
              <Icon name="download" /> {record ? (record.complete ? (record.dataGeneratedAt < manifest.generated_at ? "Download the newer data" : "Download again") : "Finish the download") : `Download ${items ? mb(total) : ""}`}
            </button>
          )}
          {state === "done" && <p className="text-[14px] font-semibold text-n-ok" role="status">Saved and verified. {pack.name} now opens without a connection.</p>}
          {state === "cancelled" && <p className="text-[14px] text-n-ink-2" role="status">Stopped. What was saved is kept; download again to finish.</p>}
          {state === "quota" && <p className="text-[14px] font-semibold text-n-down" role="alert">The device&rsquo;s storage for this site is full. Free some space, then finish the download.</p>}
          {state === "failed" && prog && <p className="text-[14px] font-semibold text-n-down" role="alert">{prog.failed.length} files could not be saved. Download again to finish; nothing saved is lost.</p>}
          <p className="text-[12.5px] leading-snug text-n-ink-3">
            {persisted ? "The browser has agreed to keep this storage." : "The browser may remove saved data when the device is low on space or the site is unused; check the pack before you leave."} Map: © OpenStreetMap contributors (ODbL), © OpenMapTiles, OpenFreeMap. Relief: NOAA NCEI.
          </p>
        </>
      )}
    </div>
  );
}

export function useRecordSource(record: PackRecord | null | undefined, name: string, online: boolean, evicted: string | null = null) {
  const setSource = useSetSource();
  useEffect(() => {
    if (record === undefined) return;
    if (!record) {
      if (evicted) setSource({ id: "pack", label: `Field pack: ${name}`, state: "unavailable", detail: "Removed by the browser; download it again for offline use", asOf: evicted });
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
  }, [record, name, online, evicted, setSource]);
}
