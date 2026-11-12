"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { MotionConfig } from "motion/react";
import { useEffect, useId, useMemo, useState, useSyncExternalStore } from "react";
import { ChartScene } from "@/nx/chart/scene";
import { Button, ButtonGroup, Dialog, Hint, IconButton, Segmented, dayTime } from "@/nx/ds";
import { Desk, DeskAside, DeskBody, DeskFoot, DeskHeader, DeskMain, PageLayout } from "@/nx/ds/layout";
import { ChartToggle } from "@/nx/ds/layout/ChartToggle";
import { LOCAL_ONLY } from "@/nx/alerts/words";
import { useFollows, useInbox } from "@/nx/alerts/useInbox";
import { exportLocalData, forgetEverything, inventory, type InventoryItem } from "@/nx/personal/forget";
import { KIND_ORDER, KIND_WORDS, type KeyKind } from "@/nx/personal/keys";
import { VESSELS, moveFishery, readPrefs, replayOpening, replayTour, setChartMode, setFisheries, setHomePort, setVessel, shownChartMode, type Prefs, type Vessel } from "@/nx/personal/prefs";
import { persistent, storeVersion, subscribe } from "@/nx/personal/store";
import type { ChartMode } from "@/nx/shell/entry";
import { useCurrentPort, useHomePortKey, usePorts, useSavedPortKey } from "@/nx/shell/port";
import { portScene } from "@/nx/shell/views";
import type { FisheryId } from "@/pilot/contracts";
import { FISHERY_OPTIONS } from "@/pilot/facts";
import s from "@/nx/alerts/alerts.module.css";

/**
 * Your data and preferences: what Neritic remembers in this browser and how it behaves, in one place. The home
 * port, the fisheries in the reader's order, the chart mode and the first-visit pieces that can be played
 * again; then everything kept on the device in plain words, a copy of it to keep, and one control that deletes
 * all of it. Nothing here is an account: there is nothing to sign in to in this preview, and the page says so.
 */
export function AccountPage({ pilotAccount }: { /** accounts are connected in this deployment: the pilot's sign-in page exists */ pilotAccount: boolean }) {
  const router = useRouter();
  const { port } = useCurrentPort();
  const ports = usePorts();
  const home = useHomePortKey();
  const saved = useSavedPortKey();
  // re-read on any change to the device's storage (this tab or another)
  const version = useSyncExternalStore(subscribe, storeVersion, () => -1);
  const [prefs, setPrefs] = useState<Prefs | null>(null);
  const [mode, setMode] = useState<ChartMode | null>(null);
  const [kept, setKept] = useState(true);
  const [said, setSaid] = useState("");
  const [deleted, setDeleted] = useState(false);
  const homeId = useId();
  const addId = useId();
  const vesselId = useId();

  // `/account#alerts` was the alerts' old address (Today's Alerts button still names it): send it on, with its port
  useEffect(() => {
    if (window.location.hash !== "#alerts") return;
    const place = new URLSearchParams(window.location.search).get("place");
    router.replace(place && /^port-\d{1,4}$/.test(place) ? `/alerts?place=${place}` : "/alerts");
  }, [router]);

  useEffect(() => {
    setPrefs(readPrefs());
    setMode(shownChartMode());
    setKept(persistent());
    setDeleted(new URLSearchParams(window.location.search).get("deleted") === "1");
  }, [version, home]);

  const name = useMemo(() => {
    const m = new Map(ports.map((p) => [p.key, p.name]));
    return (key: string | null | undefined) => (key ? (m.get(key) ?? null) : null);
  }, [ports]);
  const label = (id: FisheryId) => FISHERY_OPTIONS.find((f) => f.id === id)?.label ?? id;
  const fisheries = prefs?.fisheries ?? [];
  const rest = FISHERY_OPTIONS.filter((f) => !fisheries.includes(f.id));
  const opens = name(saved);

  return (
    <MotionConfig reducedMotion="user">
      <ChartScene scene={portScene("account", port, 30)} />
      <PageLayout mode="desk" paneKey="account" defaultPane="rail" />
      <article className={s.page} data-testid="nx-account" data-ready={prefs ? "1" : "0"}>
        <Desk>
          <DeskHeader context="Your data" title="Your data and preferences" titleProps={{ "data-testid": "account-title" }} status={<span data-testid="account-local">{LOCAL_ONLY} There is no account to sign in to in this preview.</span>} actions={<ChartToggle />} />
          <DeskBody layout="main-aside" aside="end">
            <DeskMain>
              {deleted ? (
                <Hint tone="caution">
                  <span data-testid="account-deleted">Everything Neritic kept in this browser has been deleted.</span>
                </Hint>
              ) : null}
              {!kept ? (
                <Hint tone="caution">
                  <span data-testid="account-not-kept">This browser is not keeping data (a private window, or site data is blocked). Your choices last until you close this page.</span>
                </Hint>
              ) : null}

              <section className={s.section} aria-labelledby="account-prefs">
                <h2 id="account-prefs" className={s.h2}>
                  Preferences
                </h2>

                <div className={s.pref}>
                  <label htmlFor={homeId} className={s.label}>
                    Home port
                  </label>
                  <select id={homeId} className={s.select} value={home ?? ""} onChange={(e) => setHomePort(e.target.value || null)} disabled={home === undefined} data-testid="account-home">
                    <option value="">No home port</option>
                    {ports.map((p) => (
                      <option key={p.key} value={p.key}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                  <p className={s.small} data-testid="account-opens">
                    {home === undefined ? "" : home ? `Neritic opens on ${name(home) ?? "your home port"}. Opening a link to another port does not change this.` : opens ? `With no home port, Neritic opens on the port you used last (${opens}).` : "With no home port, Neritic opens on its front page."}
                  </p>
                </div>

                <div className={s.pref}>
                  <h3 className={s.h3}>Your fisheries</h3>
                  <p className={s.small}>The first one leads Today. The order changes no value, caveat or official status.</p>
                  {fisheries.length ? (
                    <ol className={s.order} data-testid="account-fisheries">
                      {fisheries.map((f, i) => (
                        <li key={f} className={s.orderItem} data-fishery={f}>
                          <span className={s.orderName}>{label(f)}</span>
                          <span className={s.orderButtons}>
                            <IconButton icon="chevron" direction="up" size={44} label={`Move ${label(f)} up`} disabled={i === 0} onClick={() => moveFishery(f, -1)} />
                            <IconButton icon="chevron" direction="down" size={44} label={`Move ${label(f)} down`} disabled={i === fisheries.length - 1} onClick={() => moveFishery(f, 1)} />
                            <IconButton icon="close" size={44} label={`Remove ${label(f)}`} onClick={() => setFisheries(fisheries.filter((x) => x !== f))} />
                          </span>
                        </li>
                      ))}
                    </ol>
                  ) : prefs ? (
                    <p className={s.empty}>None chosen. Today asks which fishery to lead with.</p>
                  ) : null}
                  {rest.length ? (
                    <div className={s.field}>
                      <label htmlFor={addId} className={s.label}>
                        Add a fishery
                      </label>
                      <select id={addId} className={s.select} value="" onChange={(e) => e.target.value && setFisheries([...fisheries, e.target.value as FisheryId])} disabled={!prefs} data-testid="account-add-fishery">
                        <option value="">Choose a fishery</option>
                        {rest.map((f) => (
                          <option key={f.id} value={f.id}>
                            {f.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  ) : null}
                </div>

                <div className={s.pref}>
                  <label htmlFor={vesselId} className={s.label}>
                    Vessel length (optional)
                  </label>
                  <select id={vesselId} className={s.select} value={prefs?.vessel ?? ""} onChange={(e) => setVessel((e.target.value || null) as Vessel | null)} disabled={!prefs} data-testid="account-vessel">
                    <option value="">Not given</option>
                    {VESSELS.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className={s.pref}>
                  <Segmented<ChartMode>
                    label="Chart"
                    options={[
                      { value: "day", label: "Day", hint: "Sunlight" },
                      { value: "dusk", label: "Dusk", hint: "Evening wheelhouse" },
                      { value: "night", label: "Night", hint: "Dark-adapted, very dim" },
                    ]}
                    value={mode}
                    onChange={(m) => {
                      setChartMode(m);
                      setMode(m);
                    }}
                    fit="equal"
                  />
                </div>

                <div className={s.pref}>
                  <h3 className={s.h3}>First-visit guidance</h3>
                  <ButtonGroup>
                    <Button
                      variant="secondary"
                      onClick={() => {
                        replayTour();
                        setSaid("“Show me around” is offered again on Today.");
                      }}
                      data-testid="account-replay-tour"
                    >
                      Show me around again
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => {
                        replayOpening();
                        setSaid("The opening plays the next time Neritic opens on its front page.");
                      }}
                      data-testid="account-replay-opening"
                    >
                      Play the opening again
                    </Button>
                  </ButtonGroup>
                  <p className={s.status} role="status" data-testid="account-said">
                    {said}
                  </p>
                </div>
              </section>

              {pilotAccount ? (
                <section className={s.section} aria-labelledby="account-pilot">
                  <h2 id="account-pilot" className={s.h2}>
                    Pilot account
                  </h2>
                  <p className={s.lede}>
                    Pilot members can sign in with an email code to keep their ports on more than one device and to send trip feedback.{" "}
                    <a href="/account/pilot" className={s.link} data-testid="account-pilot-link">
                      Pilot sign-in and trip feedback
                    </a>
                  </p>
                </section>
              ) : null}
            </DeskMain>

            <DeskAside label="What is kept on this device">
              <h2 className={s.h2}>Kept on this device</h2>
              <DeviceData version={version} onDeleted={() => window.location.replace("/account?deleted=1")} />
            </DeskAside>
          </DeskBody>
          <DeskFoot>
            <p>
              Neritic has no copy of any of this. The one thing that leaves this browser is a question you type into Ask Neritic, which is sent to be answered.{" "}
              <Link href="/privacy" className={s.link}>
                Privacy policy
              </Link>
              {" · "}
              <Link href="/alerts" className={s.link}>
                Alerts and the ports you follow
              </Link>
            </p>
          </DeskFoot>
        </Desk>
      </article>
    </MotionConfig>
  );
}

const sizeWords = (i: { size: number; unit: InventoryItem["unit"] }) => (i.unit === "files" ? `${i.size} file${i.size === 1 ? "" : "s"}` : i.size >= 1e6 ? `${(i.size / 1e6).toFixed(1)} MB` : i.size >= 1000 ? `${Math.round(i.size / 1000)} KB` : "under 1 KB");

type Line = { holds: string; kind: KeyKind; size: number; unit: InventoryItem["unit"]; n: number };

/** The inventory as the reader's lines: one per thing kept (a per-port key is one line, however many ports). */
function lines(items: InventoryItem[]): Line[] {
  const out = new Map<string, Line>();
  for (const i of items) {
    const k = `${i.kind}|${i.holds}|${i.unit}`;
    const l = out.get(k);
    if (l) {
      l.size += i.size;
      l.n += 1;
    } else out.set(k, { holds: i.holds, kind: i.kind, size: i.size, unit: i.unit, n: 1 });
  }
  return [...out.values()];
}

function DeviceData({ version, onDeleted }: { version: number; onDeleted: () => void }) {
  const [items, setItems] = useState<InventoryItem[] | null>(null);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [said, setSaid] = useState("");
  const follows = useFollows();
  const inbox = useInbox();
  useEffect(() => {
    let live = true;
    void inventory().then((i) => live && setItems(i));
    return () => {
      live = false;
    };
  }, [version, confirm]);

  // housekeeping says nothing about the reader: it is deleted with the rest, and not listed line by line
  const shown = lines((items ?? []).filter((i) => i.kind !== "transient"));
  const housekeeping = (items ?? []).filter((i) => i.kind === "transient" && i.store !== "cache").length;

  const save = async () => {
    const data = await exportLocalData();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `neritic-my-data-${data.exportedAt.slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    setSaid(`A copy was saved as ${a.download} (${dayTime(data.exportedAt)}). Offline copies are named in it, not included.`);
  };

  const remove = async () => {
    setBusy(true);
    const gone = await forgetEverything();
    setBusy(false);
    setConfirm(false);
    if (gone.failed.length) setSaid(`${gone.failed.length} thing${gone.failed.length === 1 ? "" : "s"} could not be removed. Clearing this site's data in your browser's settings removes everything.`);
    else onDeleted();
  };

  return (
    <div data-testid="account-data" data-count={items ? shown.length : undefined}>
      {items === null ? null : !shown.length ? (
        <p className={s.empty} data-testid="account-nothing">
          Nothing about you is kept here yet.
        </p>
      ) : (
        KIND_ORDER.filter((k) => k !== "transient" && shown.some((l) => l.kind === k)).map((k) => (
          <div key={k} className={s.group}>
            <h3 className={s.h3}>{KIND_WORDS[k]}</h3>
            <ul className={s.rows} data-testid={`account-kept-${k}`}>
              {shown
                .filter((l) => l.kind === k)
                .map((l) => (
                  <li key={`${l.holds}${l.unit}`} className={s.row} data-off="">
                    <span className={s.rowLabel}>
                      {l.holds}
                      {l.n > 1 ? ` (${l.n})` : ""}
                    </span>
                    <span className={s.rowValue}>{sizeWords(l)}</span>
                  </li>
                ))}
            </ul>
          </div>
        ))
      )}
      {items && housekeeping ? <p className={s.small}>Plus {housekeeping} small housekeeping item{housekeeping === 1 ? "" : "s"} (such as whether this browser keeps data), deleted with everything else.</p> : null}
      <div className={s.actions}>
        <ButtonGroup>
          <Button variant="secondary" onClick={() => void save()} disabled={!items || !shown.length} data-testid="account-export">
            Save a copy of my data
          </Button>
          <Button variant="secondary" onClick={() => setConfirm(true)} disabled={!items} data-testid="account-delete">
            Delete everything on this device
          </Button>
        </ButtonGroup>
      </div>
      <p className={s.status} role="status" data-testid="account-data-said">
        {said}
      </p>
      <Dialog
        open={confirm}
        onOpenChange={setConfirm}
        title="Delete everything on this device?"
        footer={
          <ButtonGroup align="end">
            <Button variant="secondary" onClick={() => setConfirm(false)}>
              Keep it
            </Button>
            <Button variant="primary" onClick={() => void remove()} busy={busy} busyLabel="Deleting" data-testid="account-delete-yes">
              Delete everything
            </Button>
          </ButtonGroup>
        }
      >
        <div className={s.confirm} data-testid="account-delete-confirm">
          <p>This removes, from this browser only:</p>
          <ul>
            <li>your home port, fisheries and preferences;</li>
            <li>
              the {follows?.length ?? 0} port{follows?.length === 1 ? "" : "s"} you follow and your {inbox?.items.length ?? 0} alert{inbox?.items.length === 1 ? "" : "s"};
            </li>
            <li>every trip and vessel saved on My Coast;</li>
            <li>what each page showed on your last visit;</li>
            <li>every page saved for offline use.</li>
          </ul>
          <p>It cannot be undone. Save a copy first if you want one. Neritic then opens as it does on a first visit.</p>
        </div>
      </Dialog>
    </div>
  );
}
