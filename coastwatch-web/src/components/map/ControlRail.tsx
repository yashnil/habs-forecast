"use client";

import { useRef } from "react";
import type { Manifest } from "@/generated/schema";
import { currentsHourly } from "@/lib/currents";
import { Icon, type IconName } from "@/components/ui/Icon";
import { GROUPS, type LayerGroup } from "./LayerDock";

const GROUP_ICON: Record<LayerGroup, IconName> = { forecast: "forecast", satellite: "satellite", currents: "currents" };

/**
 * Desktop control rail (M5): a slim column on the map's left edge. Places shows or hides the
 * place card; the three layer buttons switch what the map draws. Model and observation keep
 * their colours (violet, teal) on a small marker under each label. Arrow keys move between
 * layers, as in any tab list.
 */
export function ControlRail({
  manifest,
  group,
  onGroup,
  placesOpen,
  onPlaces,
}: {
  manifest: Manifest;
  group: LayerGroup;
  onGroup: (g: LayerGroup) => void;
  placesOpen: boolean;
  onPlaces: () => void;
}) {
  const hasCurrents = currentsHourly(manifest).length > 0;
  const enabled = GROUPS.filter((g) => g.id !== "currents" || hasCurrents);
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const onKey = (e: React.KeyboardEvent) => {
    const i = enabled.findIndex((g) => g.id === group);
    const next = e.key === "ArrowDown" ? i + 1 : e.key === "ArrowUp" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? enabled.length - 1 : null;
    if (next == null) return;
    e.preventDefault();
    const g = enabled[(next + enabled.length) % enabled.length];
    onGroup(g.id);
    refs.current[g.id]?.focus();
  };
  return (
    <nav aria-label="Map controls" data-testid="control-rail" className="absolute inset-y-0 left-0 z-20 flex w-16 flex-col items-center gap-1 border-r border-white/[0.07] bg-navy-950/90 py-2.5 text-on-navy-2 backdrop-blur-sm">
      <RailButton icon="search" label="Places" active={placesOpen} onClick={onPlaces} testid="rail-places" aria-pressed={placesOpen} />
      <span className="my-1 h-px w-8 bg-white/10" aria-hidden />
      <div role="tablist" aria-label="Map layer" aria-orientation="vertical" onKeyDown={onKey} className="flex flex-col gap-1">
        {GROUPS.map((g) => {
          const off = g.id === "currents" && !hasCurrents;
          const on = group === g.id;
          return (
            <button
              key={g.id}
              ref={(el) => {
                refs.current[g.id] = el;
              }}
              type="button"
              role="tab"
              aria-selected={on}
              tabIndex={on ? 0 : -1}
              disabled={off}
              aria-disabled={off || undefined}
              title={off ? "Observed currents are not in this dataset." : `${g.label} (${g.kind.toLowerCase()})`}
              data-testid={`group-${g.id}`}
              onClick={() => onGroup(g.id)}
              className={`group relative flex w-14 flex-col items-center gap-0.5 rounded-lg pb-1.5 pt-2 text-[11px] font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-35 ${on ? "bg-white/[0.09] text-white" : "hover:bg-white/[0.05] hover:text-on-navy"}`}
            >
              {on && <span className="absolute inset-y-2 -left-1 w-[3px] rounded-r bg-accent-on-navy" aria-hidden />}
              <Icon name={GROUP_ICON[g.id]} className="h-[22px] w-[22px]" />
              {g.short}
              <span aria-hidden className={`h-[3px] w-4 rounded-full ${g.kind === "Model" ? "bg-[#b9a6f5]" : "bg-[#7fd4c1]"} ${on ? "opacity-100" : "opacity-45"}`} />
              <span className="sr-only">, {g.kind.toLowerCase()}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}

function RailButton({ icon, label, active, onClick, testid, ...rest }: { icon: IconName; label: string; active: boolean; onClick: () => void; testid: string } & React.AriaAttributes) {
  return (
    <button
      type="button"
      onClick={onClick}
      data-testid={testid}
      className={`flex w-14 flex-col items-center gap-0.5 rounded-lg pb-1.5 pt-2 text-[11px] font-medium transition-colors duration-150 ${active ? "text-white" : "hover:bg-white/[0.05] hover:text-on-navy"}`}
      {...rest}
    >
      <Icon name={icon} className="h-[22px] w-[22px]" />
      {label}
    </button>
  );
}
