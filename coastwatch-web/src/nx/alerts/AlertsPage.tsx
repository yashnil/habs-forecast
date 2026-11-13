"use client";

import Link from "next/link";
import { AnimatePresence, MotionConfig, motion } from "motion/react";
import { useEffect, useId, useMemo, useState } from "react";
import { ChartRegulation, rampZoneAt } from "@/nx/chart/regulation";
import { ChartScene } from "@/nx/chart/scene";
import { Button, Dock, FollowButton, Glyph, MapLayers, Micro, PageLayout, Ticket, TicketMain, TicketStub, ToggleRow, Disclosure, VisuallyHidden, day, dayTime, panelIn } from "@/nx/ds";
import { useOnline } from "@/nx/offline/useOffline";
import { persistent } from "@/nx/personal/store";
import { useCurrentPort, usePorts, type ShellPort } from "@/nx/shell/port";
import { portScene } from "@/nx/shell/views";
import { TRIGGERS } from "@/pilot/alerts/diff";
import { alertChange } from "./change";
import { ChangeLine } from "./ChangeLine";
import { runCheck } from "./check";
import { MAX_FOLLOWS, follow, setTrigger, unfollow, type Follow } from "./follow";
import type { InboxItem } from "./inbox";
import { useFollows, useInbox, useLastCheck } from "./useInbox";
import { CHANNELS, DELIVERY, LOCAL_ONLY, TRIGGER_WORDS, itemChange, itemTitle, lastCheckedWords, plainDates, reportWords } from "./words";
import s from "./alerts.module.css";

/**
 * Alerts (v14 synthesis, SYSTEM.md "Alerts"; journeys.md Journey 2): the chart with Regulation on, and a ticket
 * anchored to the port the alert is about. The ticket says the change: the record's words, the new date as the
 * hero with the old one struck, the before → after on a date axis ("+14 days"), "Not verified" and its source, when
 * this device noticed it; its one filled action opens the record. A dock along the foot says exactly how alerts
 * arrive (DELIVERY, verbatim: on this device, when Neritic is opened, nothing sent) with the last check, "What alerts
 * you" (the ports followed and their switches) and "Check now".
 *
 * Without an alert the ticket offers the port the reader came from ("Follow Bodega Bay"), or says the port is
 * followed and nothing has changed. Follow state is the device's (follow.ts), so Today and the chart show the same.
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
  const [picked, setPicked] = useState<string | null>(null);
  const [opened, setOpened] = useState<string | null>(null);
  const [kinds, setKinds] = useState<boolean | null>(null);
  const pickId = useId();
  const panelId = useId();
  const byKey = useMemo(() => new Map(ports.map((p) => [p.key, p])), [ports]);
  const name = (key: string) => byKey.get(key)?.name ?? "A port no longer listed";
  const sinceOf = (key: string) => follows?.find((f) => f.port === key)?.since ?? "";

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
  const items = inbox?.items ?? [];
  // the alert the ticket shows: the one the reader picked, else the newest unread, else the newest
  const lead = items.find((i) => i.key === picked) ?? items.find((i) => !i.read_at) ?? items[0] ?? null;
  const rest = lead ? items.filter((i) => i.key !== lead.key) : [];
  // "What alerts you" opens by itself when there is no alert to read (the ports and their switches are the page then)
  const kindsOpen = kinds ?? (hydrated && !lead);

  const start = (key: string) => {
    const r = follow(key);
    if (!r.ok) {
      setSaid(r.why === "full" ? `You follow ${MAX_FOLLOWS} ports, the most this device keeps. Stop following one first.` : "That is not a port Neritic lists.");
      return;
    }
    setPick("");
    setKinds(true);
    void check(true);
  };

  const last = lastCheckedWords(stamp ?? null, (iso) => dayTime(iso), name);
  const at: ShellPort | null = (lead ? byKey.get(lead.place_id) : null) ?? port;
  const zone = at ? rampZoneAt(at.lat) : null;
  const scene = portScene("alerts", at, 200);

  return (
    <MotionConfig reducedMotion="user">
      <PageLayout mode="stage" phoneMap={0.27} />
      <MapLayers page="alerts" defaults={{ regulation: true }} />
      <ChartScene scene={scene} />
      <ChartRegulation spec={{ zones: { highlight: zone } }} />
      <article className={s.page} data-testid="nx-alerts" data-ready={hydrated ? "1" : "0"}>
        <h1 className={s.vh}>What changed at the ports you follow</h1>
        <Ticket key={at?.key ?? "coast"} at={at} anchorY={176} width={472} labelledBy="alerts-title" className={s.ticket}>
          {lead ? (
            <LeadAlert item={lead} open={opened === lead.key} following={followed.has(lead.place_id)} since={sinceOf(lead.place_id)} />
          ) : offer ? (
            <TicketMain data-testid="alerts-offer" data-port={offer.key}>
              <Micro>Alerts · {offer.name}</Micro>
              <h2 id="alerts-title" className={s.offerHead}>
                Follow {offer.name}
              </h2>
              <p className={s.offerText}>Its season status, official notices, toxin tests and whale measures are noted as they are now. A change is listed here the next time you open Neritic.</p>
              <Empty following={false} />
            </TicketMain>
          ) : (
            <TicketMain>
              <Micro>{at && followed.has(at.key) ? `${at.name} · following${sinceOf(at.key) ? ` since ${day(sinceOf(at.key))}` : ""}` : "Alerts"}</Micro>
              <h2 id="alerts-title" className={s.offerHead}>
                {hydrated ? (follows?.length ? "Nothing has changed yet" : "Nothing followed yet") : " "}
              </h2>
              {hydrated ? <Empty following={(follows?.length ?? 0) > 0} /> : null}
            </TicketMain>
          )}
          <TicketStub className={s.stub}>
            {lead ? (
              <div className={s.acts}>
                <Button
                  variant="primary"
                  glyph={opened === lead.key ? "↑" : "↓"}
                  aria-expanded={opened === lead.key}
                  onClick={() => {
                    setOpened(opened === lead.key ? null : lead.key);
                    if (!lead.read_at) inbox?.markRead([lead.key]);
                  }}
                  data-testid="alerts-open"
                >
                  {opened === lead.key ? "Hide the record" : "Open the record"}
                </Button>
                <Button variant="quiet" href={`/today?place=${encodeURIComponent(lead.place_id)}`} glyph={null}>
                  Today at {lead.place_label}
                </Button>
              </div>
            ) : offer ? (
              <div className={s.acts}>
                <FollowButton name={offer.name} following={false} onChange={(on) => on && start(offer.key)} disabled={full} data-testid="alerts-follow" />
              </div>
            ) : at && followed.has(at.key) ? (
              <div className={s.acts}>
                <FollowButton variant="toggle" name={at.name} following onChange={(on) => !on && unfollow(at.key)} />
                <Button variant="quiet" href={`/today?place=${encodeURIComponent(at.key)}`} glyph={null}>
                  Today at {at.name}
                </Button>
              </div>
            ) : (
              <p className={s.small}>Choose a port under &ldquo;What alerts you&rdquo; to have it checked.</p>
            )}
            {lead && offer ? (
              <div className={s.offerRow} data-testid="alerts-offer" data-port={offer.key}>
                <div>
                  <h3 className={s.rowHead}>Follow {offer.name}</h3>
                  <p className={s.small}>Its records are noted as they are now; a change is listed here.</p>
                </div>
                <FollowButton variant="toggle" name={offer.name} following={false} onChange={(on) => on && start(offer.key)} disabled={full} data-testid="alerts-follow" />
              </div>
            ) : null}
            {items.length ? (
              <section className={s.inbox} aria-labelledby="alerts-inbox">
                <h2 id="alerts-inbox" className={s.inboxHead}>
                  <Micro>
                    {items.length === 1 ? "1 alert here" : `${items.length} alerts here`}
                    {inbox && inbox.unread ? <span data-testid="alerts-unread">{` · ${inbox.unread} unread`}</span> : null}
                  </Micro>
                </h2>
                {rest.length ? (
                  <ul className={s.items} data-testid="alerts-items">
                    {rest.map((i) => (
                      <InboxRow key={i.key} item={i} onPick={() => setPicked(i.key)} />
                    ))}
                  </ul>
                ) : null}
                <div className={s.inboxActs}>
                  {inbox && inbox.unread ? (
                    <Button variant="quiet" onClick={() => inbox.markRead("all")} data-testid="alerts-mark-read">
                      Mark all read
                    </Button>
                  ) : null}
                  <Button variant="quiet" onClick={() => inbox?.clear()} data-testid="alerts-clear">
                    Clear the list
                  </Button>
                </div>
              </section>
            ) : null}
          <AnimatePresence initial={false}>
            {kindsOpen ? (
              <motion.div key="kinds" id={panelId} className={s.panel} initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={panelIn}>
                <Ports
                  follows={follows}
                  name={name}
                  current={port?.key}
                  empty={hydrated && !offer && !follows?.length}
                />
                {hydrated ? (
                  <form
                    className={s.add}
                    onSubmit={(e) => {
                      e.preventDefault();
                      if (pick) start(pick);
                    }}
                  >
                    <label htmlFor={pickId} className={s.label}>
                      Follow another port
                    </label>
                    <span className={s.addRow}>
                      <select id={pickId} className={s.select} value={pick} onChange={(e) => setPick(e.target.value)} disabled={full} data-testid="alerts-add-port">
                        <option value="">{full ? `${MAX_FOLLOWS} ports is the most` : "Choose a port"}</option>
                        {others.map((p) => (
                          <option key={p.key} value={p.key}>
                            {p.name}
                          </option>
                        ))}
                      </select>
                      <Button variant="secondary" type="submit" disabled={!pick || full} data-testid="alerts-add">
                        Follow
                      </Button>
                    </span>
                  </form>
                ) : null}
                <h3 className={s.rowHead}>When your ports are checked</h3>
                <p className={s.small}>
                  Each time you open Neritic, this browser reads the published records for the ports you follow and compares them with what it saw last time, at most once every 30 minutes. Neritic has no server watching
                  your ports, so a change is listed the next time you open it, not the moment it happens.
                </p>
                <h3 className={s.rowHead}>How alerts reach you</h3>
                <ul className={s.channels} data-testid="alerts-channels">
                  {CHANNELS.map((c) => (
                    <li key={c.id} className={s.channel} data-off={c.available ? undefined : ""} data-channel={c.id}>
                      <span>{c.label}</span>
                      <span className={s.channelState}>{c.state}</span>
                    </li>
                  ))}
                </ul>
                <p className={s.small}>Neritic does not ask for a phone number or an email address, and has nowhere to send one.</p>
                <p className={s.small} data-testid="alerts-local">
                  {LOCAL_ONLY} The ports you follow and your alerts are not sent to Neritic or anyone else.{" "}
                  <Link href="/account" className={s.link}>
                    Your data and preferences
                  </Link>{" "}
                  shows what is kept and deletes it.
                </p>
                <p className={s.small}>
                  An alert reports a change in a record. It is not advice, and official status always comes from CDFW and CDPH.{" "}
                  <Link href={port ? `/notices?place=${encodeURIComponent(port.key)}` : "/notices"} className={s.link} data-testid="alerts-notices">
                    Every current official notice
                  </Link>
                  .
                </p>
              </motion.div>
            ) : null}
          </AnimatePresence>
          </TicketStub>
        </Ticket>

        <Dock label="How alerts reach you" width={470} className={s.dock}>
          <Micro as="h2">On this device only</Micro>
          <p className={s.delivery} data-testid="alerts-delivery">
            {DELIVERY}
          </p>
          {!kept ? (
            <p className={s.notKept} data-testid="alerts-not-kept">
              This browser is not keeping data (a private window, or site data is blocked). What you follow here lasts until you close this page.
            </p>
          ) : null}
          <div className={s.row2}>
            <span className={s.last} data-testid="alerts-last-checked">
              {stamp === undefined ? "" : (last ?? "This device has not checked yet.")}
            </span>
            <span className={s.dockActs}>
              <Button variant="quiet" onClick={() => setKinds(!kindsOpen)} aria-expanded={kindsOpen} aria-controls={panelId} data-testid="alerts-kinds">
                What alerts you
              </Button>
              <Button variant="quiet" onClick={() => void check(true)} disabled={!online || !follows?.length} busy={busy} busyLabel="Checking" data-testid="alerts-check-now">
                Check now
              </Button>
            </span>
          </div>
          <p className={s.said} role="status" data-testid="alerts-said">
            {!online ? "You're offline: nothing can be checked. The alerts already on this device are above." : said}
          </p>
        </Dock>
      </article>
    </MotionConfig>
  );
}

function Empty({ following }: { following: boolean }) {
  return (
    <p className={s.empty} data-testid="alerts-empty">
      {following ? "No alerts yet. When something you follow changes, it is listed here the next time you open Neritic." : "Nothing is being followed, so there is nothing to list. Follow a port to start."}
    </p>
  );
}

/** The ticket's alert: the record's words, the new date with the old struck, the dates on an axis, its provenance, the record. */
function LeadAlert({ item: i, open, following, since }: { item: InboxItem; open: boolean; following: boolean; since: string }) {
  const c = alertChange(i);
  const change = itemChange(i);
  const read = !!i.read_at;
  const moved = c.now && c.was && c.days !== 0;
  return (
    <TicketMain className={s.alert} data-testid="alerts-item" data-read={read ? "" : undefined} data-trigger={i.trigger} data-port={i.place_id}>
        <p className={s.kicker}>
          <Glyph kind={read ? "read" : "unread"} />
          <Micro>
            {read ? "Read" : "Unread"} · {i.place_label} · {TRIGGER_WORDS[i.trigger].label}
          </Micro>
        </p>
        <h2 id="alerts-title" className={s.what}>
          {itemTitle(i)}
        </h2>
        {c.now ? (
          <p className={s.move} aria-hidden="true">
            {moved ? (
              <>
                <span className={s.was}>{day(c.was!)}</span>
                <svg className={s.arr} viewBox="0 0 34 22">
                  <path d="M2 11h28M21 3l9 8-9 8" />
                </svg>
              </>
            ) : null}
            <span className={s.now}>{day(c.now)}</span>
            {i.sample ? (
              <span className={s.tag} data-tone="coral">
                <Glyph kind="illustrative" />
                <Micro>Sample</Micro>
              </span>
            ) : null}
          </p>
        ) : null}
        {change ? <p className={s.before}>{change}</p> : null}
        {c.now && c.was ? <ChangeLine was={c.was} now={c.now} sample={i.sample} /> : null}
        <div className={s.stamps}>
          <p className={s.stamp}>
            <Glyph kind={i.basis === "official" && i.verified === false ? "unverified" : i.basis === "forecast" ? "predicted" : "measured"} />
            <span>
              {i.source ? (
                i.source.url ? (
                  <a href={i.source.url} className={s.link} target={i.source.url.startsWith("/") ? undefined : "_blank"} rel="noreferrer">
                    {i.source.name}
                  </a>
                ) : (
                  i.source.name
                )
              ) : (
                "No source listed"
              )}
              {i.as_of ? ` · dated ${day(i.as_of)}` : ""}
              {i.basis === "official" && i.verified === false ? <span className={s.flag}>Not verified</span> : null}
              {i.basis === "forecast" ? <span className={s.flag}>Model</span> : null}
            </span>
          </p>
          <p className={s.noticed}>
            Noticed on this device {dayTime(i.created_at)}
            {following && since ? ` · following since ${day(since)}` : ""}
          </p>
          {i.sample ? (
            <p className={s.stamp}>
              <Glyph kind="illustrative" />
              <span>Sample data from Neritic&rsquo;s test records, not a published change</span>
            </p>
          ) : null}
        </div>
        <AnimatePresence initial={false}>
          {open ? (
            <motion.div key="record" className={s.record} initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={panelIn}>
              <p>{plainDates(i.body)}</p>
              {i.source?.url ? (
                <p>
                  <a href={i.source.url} className={s.link} target={i.source.url.startsWith("/") ? undefined : "_blank"} rel="noreferrer">
                    The record at {i.source.name}
                    <span aria-hidden="true"> ↗</span>
                  </a>
                </p>
              ) : null}
            </motion.div>
          ) : null}
        </AnimatePresence>
    </TicketMain>
  );
}

/** An earlier alert in the ticket's list: its state, port, kind and words; choosing it puts it in the ticket. */
function InboxRow({ item: i, onPick }: { item: InboxItem; onPick: () => void }) {
  const change = itemChange(i);
  return (
    <li className={s.item} data-read={i.read_at ? "" : undefined} data-testid="alerts-item" data-trigger={i.trigger} data-port={i.place_id}>
      <Glyph kind={i.read_at ? "read" : "unread"} label={i.read_at ? "Read" : "Unread"} className={s.itemG} />
      <div className={s.itemBody}>
        <p className={s.itemPort}>
          {i.place_label} · {TRIGGER_WORDS[i.trigger].label}
        </p>
        <h3 className={s.itemTitle}>
          <button type="button" className={s.itemPick} onClick={onPick}>
            {itemTitle(i)}
          </button>
        </h3>
        {change ? <p className={s.itemMeta}>{change}</p> : null}
        <p className={s.itemMeta}>
          {i.source ? i.source.name : null}
          {i.as_of ? ` · dated ${day(i.as_of)}` : ""} · noticed {dayTime(i.created_at)}
          {i.basis === "official" && i.verified === false ? <span className={s.flag}>Not verified</span> : null}
          {i.basis === "forecast" ? <span className={s.flag}>Model</span> : null}
          {i.sample ? <span className={s.flag}>Sample data</span> : null}
        </p>
      </div>
    </li>
  );
}

/** The ports this device follows, each with what alerts the reader there (the port the address names opens). */
function Ports({ follows, name, current, empty }: { follows: Follow[] | undefined; name: (k: string) => string; current: string | undefined; empty: boolean }) {
  if (follows && follows.length)
    return (
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
            <Disclosure summary="What alerts you" defaultOpen={f.port === current}>
              {TRIGGERS.map((t) => (
                <ToggleRow key={t} label={TRIGGER_WORDS[t].label} hint={TRIGGER_WORDS[t].hint} checked={f.triggers.includes(t)} onChange={(on) => setTrigger(f.port, t, on)} data-testid={`alerts-trigger-${f.port}-${t}`} />
              ))}
              <div className={s.portActs}>
                <FollowButton variant="toggle" name={name(f.port)} following onChange={(on) => !on && unfollow(f.port)} data-testid={`alerts-unfollow-${f.port}`} />
              </div>
            </Disclosure>
          </li>
        ))}
      </ul>
    );
  return empty ? (
    <p className={s.small} data-testid="alerts-no-follows">
      You follow no ports yet. Choose one below.
    </p>
  ) : null;
}
