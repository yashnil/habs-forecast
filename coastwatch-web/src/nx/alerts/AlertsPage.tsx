"use client";

import Link from "next/link";
import { MotionConfig } from "motion/react";
import { useEffect, useId, useMemo, useState } from "react";
import { ChartScene } from "@/nx/chart/scene";
import { Button, ButtonGroup, Disclosure, Hint, ToggleRow, VisuallyHidden, day, dayTime } from "@/nx/ds";
import { Desk, DeskAside, DeskBody, DeskFoot, DeskHeader, DeskMain, PageLayout } from "@/nx/ds/layout";
import { ChartToggle } from "@/nx/ds/layout/ChartToggle";
import { useOnline } from "@/nx/offline/useOffline";
import { persistent } from "@/nx/personal/store";
import { useCurrentPort, usePorts } from "@/nx/shell/port";
import { portScene } from "@/nx/shell/views";
import { TRIGGERS } from "@/pilot/alerts/diff";
import { runCheck } from "./check";
import { MAX_FOLLOWS, follow, setTrigger, unfollow } from "./follow";
import type { InboxItem } from "./inbox";
import { useFollows, useInbox, useLastCheck } from "./useInbox";
import { CHANNELS, DELIVERY, LOCAL_ONLY, TRIGGER_WORDS, itemChange, itemTitle, lastCheckedWords, plainDates, reportWords } from "./words";
import s from "./alerts.module.css";

/**
 * Alerts (PLAN decision 3; journeys.md Journey 2 "Alerts to subscribe"): the ports the reader follows, what
 * alerts them at each, and the alerts this device has made. A desk page: the list is the work, the chart shows
 * the port the reader came from.
 *
 * It says exactly what happens: the ports are checked in this browser when Neritic is opened, nothing is
 * checked while it is closed, and nothing is sent to a phone or an inbox. The other ways an alert could arrive
 * are listed as not available, with nothing to fill in.
 */
export function AlertsPage() {
  const { port } = useCurrentPort();
  const ports = usePorts();
  const follows = useFollows();
  const inbox = useInbox();
  const stamp = useLastCheck();
  const online = useOnline();
  const [said, setSaid] = useState("");
  const [busy, setBusy] = useState(false);
  const [kept, setKept] = useState(true);
  const [pick, setPick] = useState("");
  const pickId = useId();
  const name = useMemo(() => {
    const m = new Map(ports.map((p) => [p.key, p.name]));
    return (key: string) => m.get(key) ?? "A port no longer listed";
  }, [ports]);

  useEffect(() => setKept(persistent()), []);

  const check = async (force: boolean) => {
    setBusy(true);
    const r = await runCheck({ force });
    setBusy(false);
    setSaid(reportWords(r, name));
  };
  // opening this page is opening Neritic: a check that is due runs now (the shell's watch asks the same check)
  useEffect(() => {
    void runCheck();
  }, []);

  const hydrated = follows !== undefined && inbox !== undefined;
  const followed = new Set((follows ?? []).map((f) => f.port));
  const offer = hydrated && port && !followed.has(port.key) ? port : null;
  const full = (follows?.length ?? 0) >= MAX_FOLLOWS;
  const others = ports.filter((p) => !followed.has(p.key));

  const start = (key: string) => {
    const r = follow(key);
    if (!r.ok) {
      setSaid(r.why === "full" ? `You follow ${MAX_FOLLOWS} ports, the most this device keeps. Stop following one first.` : "That is not a port Neritic lists.");
      return;
    }
    setPick("");
    void check(true);
  };

  const last = lastCheckedWords(stamp ?? null, (iso) => dayTime(iso), name);

  return (
    <MotionConfig reducedMotion="user">
      <ChartScene scene={portScene("alerts", port, 30)} />
      <PageLayout mode="desk" paneKey="alerts" defaultPane="rail" />
      <article className={s.page} data-testid="nx-alerts" data-ready={hydrated ? "1" : "0"}>
        <Desk>
          <DeskHeader context="Alerts" title="What changed at the ports you follow" titleProps={{ "data-testid": "alerts-title" }} status={<span data-testid="alerts-delivery">{DELIVERY}</span>} actions={<ChartToggle />} />
          <DeskBody layout="main-aside" aside="end">
            <DeskMain>
              {!kept ? (
                <Hint tone="caution">
                  <span data-testid="alerts-not-kept">This browser is not keeping data (a private window, or site data is blocked). What you follow here lasts until you close this page.</span>
                </Hint>
              ) : null}
              <section className={s.section} aria-labelledby="alerts-inbox">
                <h2 id="alerts-inbox" className={s.h2}>
                  Alerts{inbox && inbox.unread ? <span data-testid="alerts-unread">{` · ${inbox.unread} unread`}</span> : null}
                </h2>
                <Inbox items={inbox?.items} following={(follows?.length ?? 0) > 0} onRead={(k) => inbox?.markRead([k])} />
                {inbox && inbox.items.length ? (
                  <div className={s.actions}>
                    <ButtonGroup>
                      {inbox.unread ? (
                        <Button variant="secondary" onClick={() => inbox.markRead("all")} data-testid="alerts-mark-read">
                          Mark all read
                        </Button>
                      ) : null}
                      <Button variant="secondary" onClick={inbox.clear} data-testid="alerts-clear">
                        Clear the list
                      </Button>
                    </ButtonGroup>
                  </div>
                ) : null}
              </section>

              <section className={s.section} aria-labelledby="alerts-when">
                <h2 id="alerts-when" className={s.h2}>
                  When your ports are checked
                </h2>
                <p className={s.lede}>
                  Each time you open Neritic, this browser reads the published records for the ports you follow and compares them with what it saw last time. It does this at most once every 30 minutes. Neritic has no server watching
                  your ports, so a change is listed the next time you open it, not the moment it happens.
                </p>
                <p className={s.status} data-testid="alerts-last-checked">
                  {stamp === undefined ? "" : (last ?? "This device has not checked yet.")}
                </p>
                <div className={s.actions}>
                  <Button variant="secondary" onClick={() => void check(true)} disabled={!online || !follows?.length} busy={busy} busyLabel="Checking" data-testid="alerts-check-now">
                    Check now
                  </Button>
                </div>
                <p className={s.status} role="status" data-testid="alerts-said">
                  {!online ? "You're offline: nothing can be checked. The alerts already on this device are above." : said}
                </p>
              </section>

              <section className={s.section} aria-labelledby="alerts-how">
                <h2 id="alerts-how" className={s.h2}>
                  How alerts reach you
                </h2>
                <ul className={s.rows} data-testid="alerts-channels">
                  {CHANNELS.map((c) => (
                    <li key={c.id} className={s.row} data-off={c.available ? undefined : ""} data-channel={c.id}>
                      <span className={s.rowLabel}>{c.label}</span>
                      <span className={s.rowValue}>{c.state}</span>
                    </li>
                  ))}
                </ul>
                <p className={s.small}>Neritic does not ask for a phone number or an email address, and has nowhere to send one.</p>
              </section>
            </DeskMain>

            <DeskAside label="Ports you follow">
              <h2 className={s.h2}>Ports you follow</h2>
              {offer ? (
                <div className={s.offer} data-testid="alerts-offer" data-port={offer.key}>
                  <h3 className={s.h3}>Follow {offer.name}</h3>
                  <p>Its season status, official notices, toxin tests and whale measures are noted as they are now. A change is listed here the next time you open Neritic.</p>
                  <Button variant="primary" onClick={() => start(offer.key)} disabled={full} data-testid="alerts-follow">
                    Follow {offer.name}
                  </Button>
                </div>
              ) : null}
              {follows && follows.length ? (
                <ul className={s.ports} data-testid="alerts-follows">
                  {follows.map((f) => (
                    <li key={f.port} className={s.port} data-testid={`alerts-port-${f.port}`}>
                      <div className={s.portHead}>
                        <span className={s.portName}>{name(f.port)}</span>
                        <Link href={`/today?place=${encodeURIComponent(f.port)}`} className={s.link}>
                          Today<VisuallyHidden> at {name(f.port)}</VisuallyHidden>
                        </Link>
                      </div>
                      <p className={s.portNote}>
                        {f.triggers.length ? `${f.triggers.length} of ${TRIGGERS.length} kinds of change alert you` : "Followed, with every alert turned off"}
                        {f.since ? ` · since ${day(f.since)}` : ""}
                      </p>
                      <Disclosure summary="What alerts you" defaultOpen={offerJustFollowed(f.port, port?.key)}>
                        {TRIGGERS.map((t) => (
                          <ToggleRow key={t} label={TRIGGER_WORDS[t].label} hint={TRIGGER_WORDS[t].hint} checked={f.triggers.includes(t)} onChange={(on) => setTrigger(f.port, t, on)} data-testid={`alerts-trigger-${f.port}-${t}`} />
                        ))}
                        <div className={s.actions}>
                          <Button variant="secondary" onClick={() => unfollow(f.port)} data-testid={`alerts-unfollow-${f.port}`}>
                            Stop following {name(f.port)}
                          </Button>
                        </div>
                      </Disclosure>
                    </li>
                  ))}
                </ul>
              ) : hydrated && !offer ? (
                <p className={s.empty} data-testid="alerts-no-follows">
                  You follow no ports yet. Choose one below.
                </p>
              ) : null}
              {hydrated ? (
                <form
                  className={s.add}
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (pick) start(pick);
                  }}
                >
                  <div className={s.field}>
                    <label htmlFor={pickId} className={s.label}>
                      Follow another port
                    </label>
                    <select id={pickId} className={s.select} value={pick} onChange={(e) => setPick(e.target.value)} disabled={full} data-testid="alerts-add-port">
                      <option value="">{full ? `${MAX_FOLLOWS} ports is the most` : "Choose a port"}</option>
                      {others.map((p) => (
                        <option key={p.key} value={p.key}>
                          {p.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <Button variant="secondary" type="submit" disabled={!pick || full} data-testid="alerts-add">
                    Follow
                  </Button>
                </form>
              ) : null}
            </DeskAside>
          </DeskBody>
          <DeskFoot>
            <p data-testid="alerts-local">
              {LOCAL_ONLY} The ports you follow and your alerts are not sent to Neritic or anyone else.{" "}
              <Link href="/account" className={s.link}>
                Your data and preferences
              </Link>{" "}
              shows what is kept and deletes it.
            </p>
            <p>
              An alert reports a change in a record. It is not advice, and official status always comes from CDFW and CDPH.{" "}
              <Link href={port ? `/notices?place=${encodeURIComponent(port.key)}` : "/notices"} className={s.link} data-testid="alerts-notices">
                Every current official notice
              </Link>
              .
            </p>
          </DeskFoot>
        </Desk>
      </article>
    </MotionConfig>
  );
}

/** The port the reader came for opens with its choices showing. */
const offerJustFollowed = (port: string, current: string | undefined) => port === current;

function Inbox({ items, following, onRead }: { items: InboxItem[] | undefined; following: boolean; onRead: (key: string) => void }) {
  if (items === undefined) return null;
  if (!items.length)
    return (
      <p className={s.empty} data-testid="alerts-empty">
        {following ? "No alerts yet. When something you follow changes, it is listed here the next time you open Neritic." : "Nothing is being followed, so there is nothing to list. Follow a port to start."}
      </p>
    );
  return (
    <ul className={s.items} data-testid="alerts-items">
      {items.map((i) => {
        const change = itemChange(i);
        return (
          <li key={i.key} className={s.item} data-read={i.read_at ? "" : undefined} data-testid="alerts-item" data-trigger={i.trigger} data-port={i.place_id}>
            <span className={s.dot} aria-hidden="true" />
            <div>
              <p className={s.itemPort}>
                {i.read_at ? null : <VisuallyHidden>Unread. </VisuallyHidden>}
                {i.place_label} · {TRIGGER_WORDS[i.trigger].label}
              </p>
              <h3 className={s.itemTitle}>{itemTitle(i)}</h3>
              {change ? <p className={s.itemChange}>{change}</p> : null}
              <p className={s.itemMeta}>
                {i.source ? (
                  i.source.url ? (
                    <a href={i.source.url} className={s.link} target={i.source.url.startsWith("/") ? undefined : "_blank"} rel="noreferrer">
                      {i.source.name}
                    </a>
                  ) : (
                    <span>{i.source.name}</span>
                  )
                ) : null}
                {i.as_of ? <span>dated {day(i.as_of)}</span> : null}
                <span>noticed {dayTime(i.created_at)}</span>
                {i.basis === "official" && i.verified === false ? <span className={s.tag}>Not verified</span> : null}
                {i.basis === "forecast" ? <span className={s.tag}>Model</span> : null}
                {i.sample ? <span className={s.tag}>Sample data</span> : null}
              </p>
              <Disclosure summary="The record" onOpenChange={(open) => open && !i.read_at && onRead(i.key)}>
                <p className={s.itemBody}>{plainDates(i.body)}</p>
                <p className={s.small}>
                  <Link href={`/today?place=${encodeURIComponent(i.place_id)}`} className={s.link}>
                    Today at {i.place_label}
                  </Link>
                </p>
              </Disclosure>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
