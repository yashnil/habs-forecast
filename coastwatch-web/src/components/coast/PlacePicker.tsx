"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import type { CoastPlace } from "@/lib/coast";

/**
 * Choose the waters: a searchable list of CDFW ports and CDPH harvest sites (combobox and
 * listbox, arrow keys, Enter, Escape). Choosing a point on the map is offered alongside.
 */
export function PlacePicker({
  places,
  current,
  onPick,
  onPickOnMap,
  label = "Choose waters",
  autoFocus = false,
}: {
  places: CoastPlace[];
  current: CoastPlace | null;
  onPick: (p: CoastPlace) => void;
  onPickOnMap?: () => void;
  label?: string;
  autoFocus?: boolean;
}) {
  const id = useId();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const shown = useMemo(() => {
    const t = q.trim().toLowerCase();
    return t ? places.filter((p) => p.name.toLowerCase().includes(t) || (p.county ?? "").toLowerCase().includes(t)) : places;
  }, [q, places]);
  useEffect(() => setActive(0), [q]);
  useEffect(() => {
    if (autoFocus) input.current?.focus();
  }, [autoFocus]);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => !box.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);

  const pick = (p: CoastPlace) => {
    onPick(p);
    setQ("");
    setOpen(false);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((a) => Math.min(shown.length - 1, a + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(0, a - 1));
    } else if (e.key === "Enter" && open && shown[active]) {
      e.preventDefault();
      pick(shown[active]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };
  const groups: { title: string; kind: CoastPlace["kind"] }[] = [
    { title: "Ports (CDFW)", kind: "port" },
    { title: "Shellfish monitoring sites (CDPH)", kind: "site" },
  ];
  let i = -1;

  return (
    <div ref={box} className="relative" data-testid="place-picker">
      <label htmlFor={`${id}-in`} className="sr-only">
        {label}
      </label>
      <div className="flex items-center gap-2 rounded-xl border border-hairline-strong bg-surface px-3 shadow-[0_1px_2px_rgba(6,17,30,0.06)] focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/25">
        <Icon name="search" className="h-[18px] w-[18px] shrink-0 text-ink-3" />
        <input
          ref={input}
          id={`${id}-in`}
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-list`}
          aria-autocomplete="list"
          aria-activedescendant={open && shown[active] ? `${id}-o-${active}` : undefined}
          value={q}
          placeholder={current ? `Change from ${current.name}` : "Search a port or harvest site"}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKey}
          className="h-12 min-w-0 flex-1 bg-transparent text-[16px] text-ink outline-none placeholder:text-ink-3"
          data-testid="place-input"
        />
        {onPickOnMap && (
          <button type="button" onClick={onPickOnMap} className="flex min-h-[40px] shrink-0 items-center gap-1.5 rounded-lg px-2 text-[14px] font-medium text-accent hover:bg-surface-3" data-testid="pick-on-map">
            <Icon name="pin" className="h-4 w-4" />
            <span className="max-sm:sr-only">Pick on the map</span>
          </button>
        )}
      </div>
      {open && (
        <div
          id={`${id}-list`}
          role="listbox"
          aria-label="Ports and harvest sites"
          className="absolute inset-x-0 top-[calc(100%+6px)] z-40 max-h-[min(420px,60vh)] overflow-y-auto rounded-xl border border-hairline bg-surface py-1.5 shadow-[0_10px_30px_rgba(6,17,30,0.18)]"
          data-testid="place-list"
        >
          {shown.length === 0 && <p className="px-3 py-2 text-[14px] text-ink-3">No port or site matches “{q}”. You can also pick a point on the map.</p>}
          {groups.map((g) => {
            const items = shown.filter((p) => p.kind === g.kind);
            if (!items.length) return null;
            return (
              <div key={g.kind} role="group" aria-label={g.title}>
                <p className="px-3 pb-1 pt-2 text-[12px] font-semibold uppercase tracking-[0.08em] text-ink-3" aria-hidden>
                  {g.title}
                </p>
                {items.map((p) => {
                  i = shown.indexOf(p);
                  const k = i;
                  return (
                    <div
                      key={p.key}
                      id={`${id}-o-${k}`}
                      role="option"
                      aria-selected={k === active}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => pick(p)}
                      onMouseEnter={() => setActive(k)}
                      className={`flex cursor-pointer items-baseline justify-between gap-3 px-3 py-2 text-[15px] ${k === active ? "bg-surface-3" : ""}`}
                      data-testid={`place-option-${p.key}`}
                    >
                      <span className="text-ink">{p.name}</span>
                      <span className="shrink-0 text-[13px] text-ink-3">{p.county ? `${p.county} County` : ""}</span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
