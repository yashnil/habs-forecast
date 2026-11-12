"use client";

import { usePathname } from "next/navigation";
import { useState } from "react";
import { Button } from "@/nx/ds/Button";
import { dayTime } from "@/nx/ds/format";
import { list, removePack, saveFieldPack, type FieldRecord, type Progress } from "./pack";
import { useFieldRecord, useOnline } from "./useOffline";
import s from "./offline.module.css";

const mb = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);

/** What a saved pack holds, in one or two plain lines. */
export function PackContents({ r }: { r: FieldRecord }) {
  // the trip worksheet is part of My Coast: one name for the reader, two saved documents
  const pages = r.pages.filter((p) => p.ok && p.id !== "trip").map((p) => p.label);
  const frames = r.frames.filter((f) => f.ok).map((f) => `${f.layer === "C-HARM" ? f.layer : f.layer.toLowerCase()} (${f.label === "Latest" ? "latest view" : f.label})`);
  return (
    <>
      <p className={s.line} data-testid="nx-pack-contents">
        {list(pages)}
        {frames.length ? `; Explore's latest frames: ${list(frames)}` : ""}. {mb(r.bytes)} on this device.
      </p>
      {r.missing.length ? <p className={s.line}>Not saved: {list(r.missing)}.</p> : null}
    </>
  );
}

/**
 * The one control that saves the reader's port for offline use (README §12), in the Menu on every page: what is
 * saved and when, save again, remove. Saving takes a few seconds and says how far it has got.
 */
export function FieldPack({ port }: { port: { key: string; name: string; lat: number; lon: number } | null }) {
  const rec = useFieldRecord();
  const online = useOnline();
  const pathname = usePathname();
  const [busy, setBusy] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const forPort = rec && port && rec.port.key === port.key ? rec : null;

  const save = async () => {
    if (!port || busy) return;
    setError(null);
    setBusy({ step: "Today" });
    const r = await saveFieldPack(port, { researchPath: pathname === "/research" ? `/research${window.location.search}` : null, onProgress: setBusy });
    setBusy(null);
    if (!r.ok) setError(`Not saved: ${r.why}`);
  };

  return (
    <section className={s.pack} aria-labelledby="nx-pack-head" data-testid="nx-pack" data-complete={forPort ? String(forPort.missing.length === 0) : undefined}>
      <h2 id="nx-pack-head" className={s.head}>
        Offline
      </h2>
      {rec === undefined ? null : !port ? (
        <p className={s.line}>Choose a port to save it for use without a connection.</p>
      ) : (
        <>
          {forPort ? (
            <>
              <p className={s.line} data-testid="nx-pack-status">
                <strong>{forPort.port.name}</strong> saved {dayTime(forPort.savedAt)}.
              </p>
              <PackContents r={forPort} />
            </>
          ) : (
            <p className={s.line}>
              Keep {port.name} on this device: Today, My Coast with your worksheet, Explore&rsquo;s latest frames for this area and Research&rsquo;s record.
              {rec ? ` Replaces the copy of ${rec.port.name}.` : ""}
            </p>
          )}
          <p className={s.actions}>
            <Button
              variant="secondary"
              onClick={save}
              disabled={!online}
              busy={!!busy}
              busyLabel={busy ? `Saving ${busy.step}` : undefined}
              data-testid="nx-pack-save"
            >
              {forPort ? "Save again" : `Save ${port.name}`}
            </Button>
            {forPort && !busy ? (
              <Button variant="text" onClick={() => void removePack()} data-testid="nx-pack-remove">
                Remove
              </Button>
            ) : null}
          </p>
          {!online ? <p className={s.line}>You&rsquo;re offline; save while you have a connection.</p> : null}
          {error ? (
            <p className={s.line} role="alert">
              {error}
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}
