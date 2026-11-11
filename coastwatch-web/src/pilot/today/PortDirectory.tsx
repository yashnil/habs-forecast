"use client";

import Link from "next/link";
import { useMemo, useRef, useState, type MouseEvent } from "react";
import { Sheet } from "@/v4/motion";
import { Icon } from "@/v4/ui/Icon";
import type { PortOption } from "./load";

/**
 * Every California port, searchable, behind one quiet action under the featured ports: closed, it takes no
 * room from the hero. Open, it is the kit Sheet (a side sheet on desktop, a bottom sheet on phones), which
 * brings the focus trap, Escape, focus return and the page's scroll lock. With a real pointer the filter takes
 * the focus at once; on a touch screen the title does, so the keyboard does not cover the list uninvited.
 */
export function PortDirectory({ ports, onOpenPort, onFollow }: { ports: PortOption[]; onOpenPort: (p: PortOption) => void; onFollow: (e: MouseEvent<HTMLAnchorElement>, p: PortOption) => void }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const filter = useRef<HTMLInputElement | null>(null);
  const rows = useMemo(() => {
    const s = q.trim().toLowerCase();
    return s ? ports.filter((p) => p.label.toLowerCase().includes(s) || (p.county ?? "").toLowerCase().includes(s)) : ports;
  }, [ports, q]);
  const close = () => {
    setOpen(false);
    setQ("");
  };
  const typeFirst = typeof window !== "undefined" && window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  return (
    <>
      <button type="button" className="today-browse nx-focus nx-press" aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen(true)} data-testid="browse-ports">
        <Icon name="search" className="h-[17px] w-[17px]" />
        Browse all California ports
      </button>
      <Sheet
        open={open}
        onClose={close}
        title="All California ports"
        testid="port-directory"
        className="today-directory"
        initialFocus={typeFirst ? filter : undefined}
        head={
          <>
            <p>The pilot ports have the most complete data; elsewhere some sections may be empty.</p>
            <label className="today-directory-filter">
              <Icon name="search" className="h-[18px] w-[18px] flex-none text-n-ink-3" />
              <span className="sr-only">Filter ports by name or county</span>
              <input
                ref={filter}
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key !== "Enter" || !rows[0]) return;
                  e.preventDefault();
                  close();
                  onOpenPort(rows[0]);
                }}
                placeholder="Filter by port or county"
                autoComplete="off"
                className="min-h-11 w-full bg-transparent text-[16px] text-n-ink outline-none placeholder:text-n-ink-3"
                data-testid="port-filter"
              />
            </label>
          </>
        }
      >
        <p className="sr-only" role="status" data-testid="port-count">
          {rows.length === 1 ? "1 port" : `${rows.length} ports`}
        </p>
        {rows.length ? (
          <ul data-testid="port-list">
            {rows.map((p) => (
              <li key={p.place_id}>
                <Link
                  href={`/today?place=${p.place_id}`}
                  onClick={(e) => {
                    close();
                    onFollow(e, p);
                  }}
                  className="today-directory-row nx-focus nx-hover-tint"
                  data-testid={`port-row-${p.place_id}`}
                >
                  <span>{p.label}</span>
                  <span className="today-directory-where">{p.pilot ? "Pilot port" : (p.county ?? "")}</span>
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <p className="today-note mt-3" data-testid="port-none">
            No port matches &ldquo;{q}&rdquo;.
          </p>
        )}
      </Sheet>
    </>
  );
}
