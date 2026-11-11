"use client";
import { Command } from "cmdk";
import { motion } from "motion/react";
import { useState } from "react";
import { Icon } from "./Icon";
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
 * Find a port by typing or by reading down the coast (spec section 9): cmdk search over CDFW landing ports,
 * a north-to-south directory grouped by region, and recent ports first. Keyboard: type, arrows, Enter.
 */
export function PortFinder({ ports, regions, recent = [], onSelect, current, label = "Find your port", autoFocus, nameLayoutId, className }: PortFinderProps) {
  const [q, setQ] = useState("");
  const byCode = new Map(ports.map((p) => [p.code, p]));
  const recents = recent.map((c) => byCode.get(c)).filter((p): p is PortEntry => Boolean(p));
  const northToSouth = [...ports].sort((a, b) => b.lat - a.lat);
  const item = (p: PortEntry, group: string) => (
    <Command.Item key={`${group}-${p.code}`} value={`${group} ${p.name} ${p.county} ${p.code}`} keywords={[p.name, p.county]} onSelect={() => onSelect(p)} className={s.item} data-current={p.code === current || undefined}>
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
    <Command className={cx(s.finder, className)} label={label} loop>
      <div className={s.inputRow}>
        <Icon name="search" />
        <Command.Input className={s.input} value={q} onValueChange={setQ} placeholder="Type a port or town" autoFocus={autoFocus} aria-label={label} />
      </div>
      <Command.List className={s.list}>
        <Command.Empty className={s.empty}>
          No CDFW landing port matches “{q}”. Try the name of the nearest town, or read down the coast below.
        </Command.Empty>
        {recents.length && !q ? (
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
      </Command.List>
      <p className={s.foot}>
        {ports.length} CDFW landing ports, north to south.
      </p>
    </Command>
  );
}
