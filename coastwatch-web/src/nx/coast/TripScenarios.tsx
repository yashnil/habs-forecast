"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Button, ButtonGroup, Drawer, EditableText, usePhone } from "@/nx/ds";
import { usePorts } from "@/nx/shell/port";
import type { CoastFishery } from "./fisheries";
import { compareTrips, exportTrips, failureWords, listTrips, shareLink, type SaveOutcome } from "./scenarios";
import { tripStore } from "./useTripStore";
import type { TripWorkspace } from "./useTripWorkspace";
import { hasEntry } from "./workspace";
import s from "./trip.module.css";

type Said = { words: string; twin?: { id: string; name: string } } | null;

/**
 * The trip's saving controls, in two parts the page places itself. `bar`: the trip's name, where it stands
 * against the saved list, and the actions that change that list (it sits in the desk header on a wide screen,
 * so it costs the results no height). `more`: the quieter actions, what was just undone or cleared, and the
 * saved trips drawer.
 *
 * What Save will do is said before it is pressed: "Update" exists only while a saved trip is open; "Save as
 * new" always adds one and refuses an exact copy (naming the trip that already holds those numbers) unless the
 * reader insists. The saved list is every port's, because the device's limit counts them all; two of them can
 * be set side by side.
 */
export function useTripScenarios({ trip, fishery, placeKey }: { trip: TripWorkspace; fishery: CoastFishery | null; placeKey: string }): { bar: ReactNode; more: ReactNode } {
  const ports = usePorts();
  const portName = useMemo(() => {
    const m = new Map(ports.map((p) => [p.key, p.name]));
    return (k: string) => m.get(k) ?? null;
  }, [ports]);
  const [said, setSaid] = useState<Said>(null);
  const [open, setOpen] = useState(false);
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [link, setLink] = useState<string | null>(null);
  // In the header (wide screens) a confirmation is said on the line kept for it, stays a few seconds and goes; a
  // question (an exact copy: open it, or save anyway?) waits for its answer. On a phone the controls sit in the
  // page, where a line that vanished would move everything under the reader's thumb: there it stays until the
  // next action replaces it.
  const phone = usePhone();
  useEffect(() => {
    if (!said || said.twin || phone) return;
    const t = window.setTimeout(() => setSaid(null), 6000);
    return () => window.clearTimeout(t);
  }, [said, phone]);
  const rows = useMemo(() => listTrips(trip.data, { placeKey, portName }), [trip.data, placeKey, portName]);
  const st = trip.state;
  const entry = hasEntry(trip.ws);
  const isOpen = st.kind === "saved" || st.kind === "changed";

  const tell = (r: SaveOutcome, done: (name: string) => string) => {
    if (r.ok) setSaid({ words: done(r.name) });
    else if (r.reason === "identical") setSaid({ words: failureWords("identical", r.name), twin: { id: r.id, name: r.name } });
    else setSaid({ words: failureWords(r.reason) });
  };
  const copyLink = async () => {
    const url = shareLink(window.location.origin, trip.ws, fishery);
    try {
      await navigator.clipboard.writeText(url);
      setLink(null);
      setSaid({ words: "Link copied. It carries the numbers only: no port, no trip name, nothing about you." });
    } catch {
      // no clipboard here: the link is shown to copy by hand
      setLink(url);
      setSaid({ words: "Copy the link below. It carries the numbers only: no port, no trip name, nothing about you." });
    }
  };
  const download = () => {
    const blob = new Blob([exportTrips(tripStore())], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "neritic-trips.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const a = picked.length === 2 ? trip.data?.scenarios.find((x) => x.id === picked[0]) : undefined;
  const b = picked.length === 2 ? trip.data?.scenarios.find((x) => x.id === picked[1]) : undefined;
  const cmp = a && b ? compareTrips(a, b, portName) : null;

  const bar = (
    <section className={s.scBar} aria-label="This trip" data-testid="trip-strip">
      <div className={s.stripHead}>
        {isOpen ? (
          <span className={s.nameOne} title={st.name}>
            <EditableText label="Trip name" value={st.name} maxLength={90} size="body" onChange={(name) => tell(trip.rename(st.id, name), (n) => `Renamed to “${n}”.`)} data-testid="trip-name" />
          </span>
        ) : (
          <p className={s.tripName} data-testid="trip-name">
            {entry ? "A trip not saved yet" : "A new trip"}
          </p>
        )}
        <p className={s.tripState} data-state={st.kind} data-testid="trip-state">
          {st.words}
        </p>
      </div>
      <ButtonGroup wrap={false} className={s.scButtons}>
        {isOpen ? (
          <Button variant={st.kind === "changed" ? "primary" : "secondary"} disabled={st.kind !== "changed"} onClick={() => tell(trip.update(), (n) => `Updated “${n}”. Kept on this device only.`)} data-testid="trip-update">
            Update
          </Button>
        ) : null}
        <Button variant={isOpen ? "secondary" : "primary"} disabled={!entry} onClick={() => tell(trip.saveAsNew(), (n) => `Saved as “${n}”. Kept on this device only.`)} data-testid="trip-save-new">
          Save as new
        </Button>
        <Button variant="secondary" onClick={() => setOpen(true)} data-testid="trip-saved-open">
          Saved trips ({trip.capacity.used})
        </Button>
      </ButtonGroup>
      <p className={s.said} aria-live="polite" data-testid="trip-said" title={said?.words}>
        {said?.twin ? (
          <>
            {/* the whole sentence for a screen reader; the short one for the line it has */}
            <span className={s.vh}>{said.words}</span>
            <span aria-hidden="true">Already saved with these exact numbers.</span>{" "}
            <Button
              variant="quiet"
              className={s.inlineButton}
              onClick={() => {
                trip.open(said.twin!.id);
                setSaid({ words: `Opened “${said.twin!.name}”.` });
              }}
            >
              Open it
            </Button>{" "}
            <Button variant="quiet" className={s.inlineButton} onClick={() => tell(trip.saveAsNew({ allowIdentical: true }), (n) => `Saved a second copy as “${n}”.`)} data-testid="trip-save-anyway">
              Save a copy anyway
            </Button>
          </>
        ) : (
          (said?.words ?? "")
        )}
      </p>
    </section>
  );

  const more = (
    <section className={s.scMore} aria-label="More for this trip, and your saved trips">
      <p className={s.said}>Trips are kept on this device only. Nothing is sent to Neritic.</p>
      <div className={s.scMoreActions}>
        {isOpen ? (
          <Button variant="quiet" className={s.compact} onClick={() => tell(trip.duplicate(st.id), (n) => `Copied as “${n}”.`)} data-testid="trip-duplicate">
            Duplicate this trip
          </Button>
        ) : null}
        <Button variant="quiet" className={s.compact} disabled={!entry} onClick={copyLink} data-testid="trip-copy-link">
          Copy numbers-only link
        </Button>
        {entry || isOpen ? (
          <Button
            variant="quiet"
            className={s.compact}
            onClick={() => {
              trip.clear();
              setSaid({ words: isOpen ? "Started a new trip. The saved one is unchanged." : "Cleared." });
            }}
            data-testid="trip-clear"
          >
            {isOpen ? "New trip" : "Clear"}
          </Button>
        ) : null}
      </div>
      {link ? <input className={s.linkBox} readOnly value={link} aria-label="Numbers-only link" onFocus={(e) => e.target.select()} data-testid="trip-link" /> : null}
      {trip.cleared.length ? (
        <p className={s.said} data-testid="trip-cleared">
          Changing fishery cleared the {trip.cleared.join(", ")}: {trip.cleared.length === 1 ? "it" : "they"} described the other fishery.
        </p>
      ) : null}
      {trip.undo ? (
        <p className={s.said} data-testid="trip-undo">
          Deleted “{trip.undo.name}”.{" "}
          <Button variant="quiet" className={s.inlineButton} onClick={() => trip.undo!.run()}>
            Undo
          </Button>
        </p>
      ) : null}

      <Drawer
        title="Saved trips"
        description={`${trip.capacity.words} Kept on this device only.`}
        open={open}
        onOpenChange={(o) => {
          setOpen(o);
          if (!o) {
            setRenaming(null);
            setPicked([]);
          }
        }}
        footer={
          <ButtonGroup>
            <Button variant="secondary" onClick={download} disabled={!rows.length} data-testid="trip-export">
              Export as a file (JSON)
            </Button>
          </ButtonGroup>
        }
      >
        {trip.undo ? (
          <p className={s.said} data-testid="trip-undo-drawer">
            Deleted “{trip.undo.name}”.{" "}
            <Button variant="quiet" className={s.inlineButton} onClick={() => trip.undo!.run()}>
              Undo
            </Button>
          </p>
        ) : null}
        {rows.length === 0 ? (
          <p className={s.said}>No trips are saved on this device yet. Fill in a trip and choose Save as new.</p>
        ) : (
          <ul className={s.saved} aria-label="Trips saved on this device">
            {rows.map((r) => (
              <li key={r.id} data-testid="trip-saved-row" data-open={trip.openId === r.id || undefined}>
                {renaming?.id === r.id ? (
                  <form
                    className={s.rename}
                    onSubmit={(e) => {
                      e.preventDefault();
                      const out = trip.rename(r.id, renaming.name);
                      if (out.ok) setRenaming(null);
                      else setSaid({ words: failureWords(out.reason) });
                    }}
                  >
                    <input className={s.nameInput} value={renaming.name} onChange={(e) => setRenaming({ id: r.id, name: e.target.value })} aria-label={`New name for ${r.name}`} maxLength={90} autoFocus />
                    <Button variant="secondary" type="submit" disabled={!renaming.name.trim()}>
                      Save name
                    </Button>
                    <Button variant="quiet" onClick={() => setRenaming(null)}>
                      Cancel
                    </Button>
                  </form>
                ) : (
                  <>
                    <p className={s.savedName}>
                      {r.name}
                      {trip.openId === r.id ? <span className={s.savedOpen}> · open now</span> : null}
                    </p>
                    <p className={s.savedMeta}>{[r.port ?? "A port not on this device's list", r.fishery, r.detail, r.outcome, `saved ${r.savedAt}`].filter(Boolean).join(" · ")}</p>
                    <div className={s.savedActions}>
                      <Button
                        variant="quiet"
                        onClick={() => {
                          trip.open(r.id);
                          setOpen(false);
                          setSaid({ words: r.here ? `Opened “${r.name}”.` : `Opened a copy of “${r.name}” here. It was saved at ${r.port ?? "another port"}; Save as new keeps it for this port.` });
                        }}
                        aria-label={`Open ${r.name}`}
                      >
                        Open
                      </Button>
                      <Button variant="quiet" onClick={() => trip.duplicate(r.id)} aria-label={`Duplicate ${r.name}`} disabled={trip.capacity.full}>
                        Duplicate
                      </Button>
                      <Button variant="quiet" onClick={() => setRenaming({ id: r.id, name: r.name })} aria-label={`Rename ${r.name}`}>
                        Rename
                      </Button>
                      <Button
                        variant="quiet"
                        onClick={() => {
                          trip.remove(r.id);
                          setPicked((p) => p.filter((x) => x !== r.id));
                        }}
                        aria-label={`Delete ${r.name}`}
                      >
                        Delete
                      </Button>
                      <label className={s.pick}>
                        <input type="checkbox" checked={picked.includes(r.id)} disabled={!picked.includes(r.id) && picked.length >= 2} onChange={(e) => setPicked((p) => (e.target.checked ? [...p, r.id] : p.filter((x) => x !== r.id)))} />
                        <span>
                          Compare<span className={s.vh}> {r.name}</span>
                        </span>
                      </label>
                    </div>
                  </>
                )}
              </li>
            ))}
          </ul>
        )}
        {rows.length > 1 && !cmp ? <p className={s.said}>Tick Compare on two trips to see only what differs between them.</p> : null}
        {cmp ? (
          <section className={s.compare} aria-label="The two trips compared" data-testid="trip-compare">
            <h3 className={s.h3}>Only what differs</h3>
            <p className={s.said}>
              <strong>A</strong> is “{cmp.a.name}”. <strong>B</strong> is “{cmp.b.name}”. Change is B less A.
            </p>
            <table className={s.cmpTable}>
              <thead>
                <tr>
                  <td />
                  <th scope="col">A</th>
                  <th scope="col">B</th>
                  <th scope="col">Change</th>
                </tr>
              </thead>
              <tbody>
                {cmp.inputs.length === 0 ? (
                  <tr>
                    <td colSpan={4}>The two trips have the same inputs.</td>
                  </tr>
                ) : (
                  cmp.inputs.map((i) => (
                    <tr key={i.label} data-kind="input">
                      <th scope="row">{i.label}</th>
                      <td>{i.a}</td>
                      <td>{i.b}</td>
                      <td />
                    </tr>
                  ))
                )}
                {cmp.outputs.map((o) => (
                  <tr key={o.id} data-kind="output" data-net={o.id === "netToBoat" || undefined}>
                    <th scope="row">{o.label}</th>
                    <td>{o.a}</td>
                    <td>{o.b}</td>
                    <td>{o.deltaWords ?? "no value"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!cmp.comparable ? <p className={s.said}>A commercial trip and a recreational trip share no result to set side by side: one has a net to boat, the other only a cost.</p> : null}
          </section>
        ) : null}
      </Drawer>
    </section>
  );

  return { bar, more };
}
