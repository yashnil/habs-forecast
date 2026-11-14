"use client";
import { Command } from "cmdk";
import { motion } from "motion/react";
import { useState } from "react";
import { Icon } from "./Icon";
import { rankPorts } from "./portSearch";
import { cx } from "./util";
import s from "./PortFinder.module.css";

export interface PortEntry {
  code: number;
  name: string;
  county: string;
  region: string;
  lat: number;
}

export interface PortFinderProps {
  ports: PortEntry[];
  /** Regions in north-to-south order, with their labels. */
  regions: Array<{ id: string; label: string }>;
  /** Port codes chosen before, most recent first. */
  recent?: number[];
  onSelect: (port: PortEntry) => void;
  /** The port currently chosen, marked in the list. */
  current?: number;
  label?: string;
  autoFocus?: boolean;
  /**
   * A Motion `layoutId` for a port's name, so the name can travel on into the next view's title (spec
   * section 8, `layout`). Give it to the chosen port only; every other name returns undefined.
   */
  nameLayoutId?: (port: PortEntry) => string | undefined;
  className?: string;
}

/**
 * Find a port by typing or by reading down the coast (spec section 9). With nothing typed: recent ports first,
 * then a north-to-south directory grouped by region. While typing: one flat list ranked by `rankPorts` (port
 * name first, county and code as keywords), the best match selected, so Enter opens it. No match is said in
 * words. cmdk supplies the keyboard (type, arrows, Enter) but not the matching: its fuzzy filter is off.
 */
const NONE = "none";

export function PortFinder({ ports, regions, recent = [], onSelect, current, label = "Find your port", autoFocus, nameLayoutId, className }: PortFinderProps) {
  const [q, setQ] = useState("");
  // nothing is selected until the reader types or presses an arrow: cmdk would otherwise select the first port
  // and scroll it into view on mount, scrolling the whole page to wherever the finder sits (/system, critic 18)
  const [selected, setSelected] = useState(NONE);
  const byCode = new Map(ports.map((p) => [p.code, p]));
  const recents = recent.map((c) => byCode.get(c)).filter((p): p is PortEntry => Boolean(p));
  const northToSouth = [...ports].sort((a, b) => b.lat - a.lat);
  const typing = q.trim().length > 0;
  const ranked = typing ? rankPorts(ports, q) : [];
  const valueOf = (p: PortEntry, group: string) => `${group}-${p.code}`;
  const onQuery = (next: string) => {
    setQ(next);
    // the best match is the selection, so Enter opens what was typed (cmdk would keep a stale one)
    const top = next.trim() ? rankPorts(ports, next)[0] : undefined;
    setSelected(top ? valueOf(top, "match") : NONE);
  };
  const item = (p: PortEntry, group: string) => (
    <Command.Item key={valueOf(p, group)} value={valueOf(p, group)} onSelect={() => onSelect(p)} className={s.item} data-current={p.code === current || undefined}>
      <Icon name="port" size={20} />
      {nameLayoutId?.(p) ? (
        <motion.span layoutId={nameLayoutId(p)} className={s.name}>
          {p.name}
        </motion.span>
      ) : (
        <span className={s.name}>{p.name}</span>
      )}
      <span className={s.county}>{p.county} County</span>
      {p.code === current ? <span className={s.currentWord}>Chosen</span> : null}
    </Command.Item>
  );
  return (
    <Command className={cx(s.finder, className)} label={label} loop shouldFilter={false} value={selected} onValueChange={setSelected}>
      <div className={s.inputRow}>
        <Icon name="search" />
        <Command.Input className={s.input} value={q} onValueChange={onQuery} placeholder="Type a port or town" autoFocus={autoFocus} aria-label={label} />
      </div>
      <Command.List className={s.list}>
        {typing ? (
          ranked.map((p) => item(p, "match"))
        ) : (
          <>
            {recents.length ? (
              <Command.Group heading="Recent" className={s.group}>
                {recents.map((p) => item(p, "recent"))}
              </Command.Group>
            ) : null}
            {regions.map((r) => {
              const inR = northToSouth.filter((p) => p.region === r.id);
              if (!inR.length) return null;
              return (
                <Command.Group key={r.id} heading={r.label} className={s.group}>
                  {inR.map((p) => item(p, r.id))}
                </Command.Group>
              );
            })}
          </>
        )}
      </Command.List>
      {/* outside the listbox (only options belong in it); a live region, so the empty result is announced */}
      <div role="status" aria-live="polite">
        {typing && !ranked.length ? (
          <p className={s.empty} data-testid="nx-port-finder-empty">
            No CDFW landing port matches “{q.trim()}”. Try the name of the nearest town, or clear the search to read down the coast.
          </p>
        ) : null}
      </div>
      <p className={s.foot}>
        {typing ? `${ranked.length} of ${ports.length} CDFW landing ports, best match first.` : `${ports.length} CDFW landing ports, north to south.`}
      </p>
    </Command>
  );
}
