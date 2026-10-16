"use client";

import { useRef, useState } from "react";
import { motion } from "motion/react";
import { AUDIENCES, type Audience, type CoastPlace } from "@/lib/coast";
import { SPECIES, type Species, type SpeciesId } from "@/content/species";
import type { TripDay } from "@/lib/trip";
import { PlacePicker } from "@/components/coast/PlacePicker";
import { Sheet } from "./Sheet";

/** species names that fit a third of a phone's width */
const SHORT_SPECIES: Partial<Record<SpeciesId, string>> = {
  any: "Any",
  dungeness: "Dungeness",
  "razor-clam": "Razor clam",
  "anchovy-sardine": "Anchovy",
  bivalve: "Clams",
  groundfish: "Rockfish",
  halibut: "Halibut",
  squid: "Squid",
  lobster: "Lobster",
};

type Which = "place" | "activity" | "species" | "day" | null;

const ACTIVITY: Record<Audience, { label: string; hint: string }> = {
  commercial: {
    label: "Commercial",
    hint: "Seasons, closures and permits for commercial fisheries",
  },
  recreational: {
    label: "Recreational",
    hint: "Sport fishing rules, regions and closures",
  },
  shellfish: {
    label: "Shellfish harvest",
    hint: "CDPH health advisories and shellfish testing first",
  },
  all: { label: "Any activity", hint: "Every notice for these waters" },
};

/**
 * "Your next trip": the one thing the page asks for, as a sentence of four answers (where, what
 * kind of fishing, which species, which day). Each answer opens its own picker; the briefing
 * below follows every change at once. Nothing is required: a place alone gives a general briefing.
 */
export function TripBar({
  place,
  places,
  onPlace,
  audience,
  onAudience,
  species,
  onSpecies,
  days,
  date,
  onDate,
  home,
  onPickOnMap,
  action,
}: {
  /** beside the place name (copy link) */
  action?: React.ReactNode;
  place: CoastPlace;
  places: CoastPlace[];
  onPlace: (p: CoastPlace) => void;
  audience: Audience;
  onAudience: (a: Audience) => void;
  species: Species;
  onSpecies: (s: SpeciesId) => void;
  days: TripDay[];
  date: string | null;
  onDate: (d: string) => void;
  home: CoastPlace | null;
  onPickOnMap?: () => void;
}) {
  const [open, setOpen] = useState<Which>(null);
  const refs = {
    place: useRef<HTMLButtonElement>(null),
    activity: useRef<HTMLButtonElement>(null),
    species: useRef<HTMLButtonElement>(null),
    day: useRef<HTMLButtonElement>(null),
  };
  const day = days.find((d) => d.date === date) ?? days[0];
  const speciesOptions = SPECIES.filter((s) => s.id === "any" || audience === "all" || s.activities.includes(audience));
  const close = () => setOpen(null);

  return (
    <section aria-label="Your next trip" className="relative" data-testid="trip-bar">
      <p className="text-[13px] font-medium text-accent">Your next trip</p>
      <div className="mt-1 flex items-start justify-between gap-3">
        <div className="relative min-w-0 flex-1">
          <motion.h1
            key={place.key}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.32, ease: [0.2, 0, 0, 1] }}
            className="font-display text-[34px] font-medium leading-[1.05] text-ink sm:text-[44px]"
           
          >
            <button
              ref={refs.place}
              type="button"
              onClick={() => setOpen(open === "place" ? null : "place")}
              aria-expanded={open === "place"}
              aria-haspopup="dialog"
              className="cw-press group -ml-1.5 inline-flex min-h-[44px] items-center rounded-xl px-1.5 text-left hover:bg-surface-3/70"
              data-testid="trip-place"
            >
              <span data-testid="coast-title">{place.name}</span>
              <span aria-hidden className="ml-2 inline-block align-middle text-[0.45em] text-ink-3 transition-transform group-hover:translate-y-0.5">
                ▾
              </span>
              <span className="sr-only">, change waters</span>
            </button>
          </motion.h1>
          <Sheet open={open === "place"} onClose={close} title="Where are you going?" testid="trip-sheet-place" returnFocus={refs.place}>
            <PlacePicker
              places={places}
              current={place}
              onPick={(p) => {
                onPlace(p);
                close();
              }}
              label="Search a port or harvest site"
              autoFocus
            />
            <div className="mt-3 flex flex-wrap gap-2 text-[14px]">
              {home && home.key !== place.key && (
                <button
                  type="button"
                  onClick={() => (onPlace(home), close())}
                  className="cw-press min-h-[44px] rounded-full border border-hairline-strong px-4 font-medium text-ink hover:border-ink-3"
                  data-testid="trip-home"
                >
                  Home port: {home.name}
                </button>
              )}
              {onPickOnMap && (
                <button
                  type="button"
                  onClick={() => (onPickOnMap(), close())}
                  className="cw-press min-h-[44px] rounded-full border border-hairline-strong px-4 font-medium text-ink hover:border-ink-3"
                >
                  Tap the water on the map
                </button>
              )}
            </div>
          </Sheet>
        </div>
        {action}
      </div>
      <p className="mt-1 text-[14px] text-ink-3" data-testid="coast-subtitle">
        {place.note}
        {place.county ? ` · ${place.county} County` : ""}
      </p>

      <div className="mt-4 grid grid-cols-3 gap-1.5 sm:flex sm:flex-wrap sm:gap-2" role="group" aria-label="Trip details">
        <div className="relative min-w-0">
          <Chip
            label="Activity"
            value={ACTIVITY[audience].label}
            short={{ all: "Any", commercial: "Commercial", recreational: "Recreational", shellfish: "Shellfish" }[audience]}
            on={open === "activity"}
            onClick={() => setOpen(open === "activity" ? null : "activity")}
            btn={refs.activity}
            testid="trip-activity"
          />
          <Sheet open={open === "activity"} onClose={close} title="What kind of trip?" testid="trip-sheet-activity" returnFocus={refs.activity}>
            <Options
              items={AUDIENCES.map((a) => a.id)
                .sort((a, b) => Number(a === "all") - Number(b === "all"))
                .map((id) => ({
                  id,
                  label: ACTIVITY[id].label,
                  hint: ACTIVITY[id].hint,
                }))}
              value={audience}
              onPick={(a) => {
                onAudience(a as Audience);
                close();
              }}
              testid="trip-for"
            />
          </Sheet>
        </div>
        <div className="relative min-w-0">
          <Chip
            label="Species"
            value={species.id === "any" ? "Any species" : species.label}
            short={SHORT_SPECIES[species.id] ?? species.label}
            on={open === "species"}
            onClick={() => setOpen(open === "species" ? null : "species")}
            btn={refs.species}
            testid="trip-species"
            prompt={species.id === "any"}
          />
          <Sheet open={open === "species"} onClose={close} title="Which species?" testid="trip-sheet-species" returnFocus={refs.species}>
            <div role="radiogroup" aria-label="Species" className="grid grid-cols-2 gap-2">
              {speciesOptions.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  role="radio"
                  aria-checked={species.id === s.id}
                  onClick={() => {
                    onSpecies(s.id as SpeciesId);
                    close();
                  }}
                  className={`cw-press min-h-[52px] rounded-xl border px-3 text-left text-[15px] font-medium transition-colors ${species.id === s.id ? "border-navy-900 bg-navy-900 text-white" : "border-hairline-strong bg-surface text-ink hover:border-ink-3"}`}
                  data-testid={`trip-species-${s.id}`}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <p className="mt-3 text-[13px] text-ink-3">The species picks its season status, the notices that name it, and its rules.</p>
          </Sheet>
        </div>
        <div className="relative min-w-0">
          <Chip label="Day" value={day ? `${day.label}, ${day.day}` : "Today"} short={day ? (day.label === "Today" || day.label === "Tomorrow" ? day.label : `${day.label} ${day.day.replace(/^\w+ /, "")}`) : "Today"} on={open === "day"} onClick={() => setOpen(open === "day" ? null : "day")} btn={refs.day} testid="trip-day" />
          <Sheet open={open === "day"} onClose={close} title="When are you going?" testid="trip-sheet-day" returnFocus={refs.day}>
            <div role="radiogroup" aria-label="Trip day" className="grid gap-1.5">
              {days.map((d) => (
                <button
                  key={d.date}
                  type="button"
                  role="radio"
                  aria-checked={d.date === date}
                  onClick={() => {
                    onDate(d.date);
                    close();
                  }}
                  className={`cw-press flex min-h-[48px] items-center justify-between gap-3 rounded-xl border px-3.5 text-left transition-colors ${d.date === date ? "border-navy-900 bg-navy-900 text-white" : "border-hairline bg-surface text-ink hover:border-ink-3"}`}
                  data-testid={`trip-day-${d.date}`}
                >
                  <span className="text-[15px] font-medium">
                    {d.label} <span className="font-normal opacity-75">· {d.day}</span>
                  </span>
                  <span className={`text-[12.5px] ${d.date === date ? "text-white/80" : "text-ink-3"}`}>{d.lead != null ? "Model outlook · tides" : "Tides only"}</span>
                </button>
              ))}
            </div>
            <p className="mt-3 text-[13px] leading-snug text-ink-3">
              NOAA&apos;s bloom model covers only the days of its latest run. On a day it does not reach, the briefing shows the notices, the latest samples and the tides, and says the model does not
              reach it.
            </p>
          </Sheet>
        </div>
      </div>
    </section>
  );
}

function Chip({
  label,
  value,
  on,
  onClick,
  btn,
  testid,
  prompt = false,
  short,
}: {
  /** the value on a phone, where three chips share one row */
  short?: string;
  label: string;
  value: string;
  on: boolean;
  onClick: () => void;
  btn: React.RefObject<HTMLButtonElement | null>;
  testid: string;
  prompt?: boolean;
}) {
  return (
    <button
      ref={btn}
      type="button"
      onClick={onClick}
      aria-expanded={on}
      aria-haspopup="dialog"
      className={`cw-press flex min-h-[52px] w-full min-w-0 items-center justify-between gap-1 rounded-2xl border px-2.5 text-left sm:w-auto sm:gap-2 sm:px-3.5 transition-[border-color,background-color,box-shadow] duration-150 ${
        on
          ? "border-navy-900 bg-surface shadow-[0_0_0_3px_rgba(11,106,133,0.18)]"
          : prompt
            ? "border-dashed border-accent/70 bg-surface hover:border-accent"
            : "border-hairline-strong bg-surface hover:border-ink-3"
      }`}
      data-testid={testid}
    >
      <span className="flex min-w-0 flex-col leading-tight">
        <span className="text-[11.5px] font-medium text-ink-3">{label}</span>
        <span className={`truncate text-[15px] font-semibold ${prompt ? "text-accent" : "text-ink"}`}>
          {short ? (
            <>
              <span className="sm:hidden">{short}</span>
              <span className="max-sm:hidden">{value}</span>
            </>
          ) : (
            value
          )}
        </span>
      </span>
      <span aria-hidden className={`text-[12px] text-ink-3 max-[359px]:hidden transition-transform duration-200 ${on ? "rotate-180" : ""}`}>
        ▾
      </span>
    </button>
  );
}

function Options({ items, value, onPick, testid }: { items: { id: string; label: string; hint: string }[]; value: string; onPick: (id: string) => void; testid: string }) {
  return (
    <div role="radiogroup" className="grid gap-1.5">
      {items.map((x) => (
        <button
          key={x.id}
          type="button"
          role="radio"
          aria-checked={x.id === value}
          onClick={() => onPick(x.id)}
          className={`cw-press min-h-[56px] rounded-xl border px-3.5 py-2 text-left transition-colors ${x.id === value ? "border-navy-900 bg-navy-900 text-white" : "border-hairline bg-surface text-ink hover:border-ink-3"}`}
          data-testid={`${testid}-${x.id}`}
        >
          <span className="block text-[15px] font-semibold">{x.label}</span>
          <span className={`block text-[13px] ${x.id === value ? "text-white/80" : "text-ink-3"}`}>{x.hint}</span>
        </button>
      ))}
    </div>
  );
}
