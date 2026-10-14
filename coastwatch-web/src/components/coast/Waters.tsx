"use client";

import Link from "next/link";
import { AnimatePresence, motion } from "motion/react";
import type { LayerArtifact } from "@/generated/schema";
import type { StationLite } from "@/lib/data";
import type { Audience } from "@/lib/coast";
import type { Species, SpeciesId } from "@/content/species";
import { ageWords, fmtMeasured, MEASURED_VARIABLES, RECENT_DAYS } from "@/lib/hab";
import { formatDate } from "@/lib/time";
import { Mark } from "@/components/ui/Mark";
import { TargetPicker } from "./Briefing";

/**
 * "Your waters and species": the kind of fishing, the species (encouraged, never assumed: skipped,
 * the page is a general overview) and, optionally, the trip day, which picks the model day and the
 * marine forecast period shown below.
 */
export function WatersAndSpecies({
  audience,
  onAudience,
  species,
  onSpecies,
  leads,
  lead,
  onLead,
  today,
}: {
  today: string | null;
  audience: Audience;
  onAudience: (a: Audience) => void;
  species: Species;
  onSpecies: (s: SpeciesId) => void;
  leads: (LayerArtifact | null)[];
  lead: number;
  onLead: (l: number) => void;
}) {
  return (
    <section aria-labelledby="waters-h" className="rounded-[18px] border border-hairline bg-surface px-4 py-4 sm:px-5" data-testid="coast-waters">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id="waters-h" className="font-display text-[20px] font-medium leading-snug text-ink">
          Your waters and species
        </h2>
        <label className="inline-flex min-h-[36px] items-center gap-2 rounded-lg border border-hairline-strong px-2.5 text-[13px] text-ink-3">
          Trip day
          <select value={lead} onChange={(e) => onLead(Number(e.target.value))} className="min-h-[32px] bg-transparent font-medium text-ink outline-none" data-testid="coast-day">
            {leads.map((l, i) =>
              l?.time.valid_date && (!today || !leads.some((x) => x?.time.valid_date && x.time.valid_date >= today) || l.time.valid_date >= today) ? (
                <option key={i} value={i}>
                  {l.time.valid_date === today ? `Today, ${formatDate(l.time.valid_date).replace(/^\w+, /, "")}` : formatDate(l.time.valid_date)}
                </option>
              ) : null,
            )}
          </select>
        </label>
      </div>
      <div className="mt-3">
        <TargetPicker audience={audience} onAudience={onAudience} species={species} onSpecies={onSpecies} compact />
      </div>
      <AnimatePresence initial={false}>
        {species.id === "any" && (
          <motion.p
            key="prompt"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden text-[14px] leading-snug text-ink-2"
            data-testid="species-prompt"
          >
            <span className="mt-3 block">Choose a species to see the notices that name it, its rules, steps and landings history. Without one, this is a general overview.</span>
          </motion.p>
        )}
      </AnimatePresence>
    </section>
  );
}

/** A station chosen on the map: its latest measurements, each with its own date, and a link to its history. */
export function StationPeek({ station, km, now, onClose }: { station: StationLite; km: number; now: Date; onClose: () => void }) {
  const rows = [...MEASURED_VARIABLES.cells, ...MEASURED_VARIABLES.toxin, ...MEASURED_VARIABLES.chl]
    .map((v) => {
      const r = station.recent?.[v]?.at(-1);
      const l = station.latest.find((x) => x.variable === v && x.value != null);
      const date = r && (!l || r[0] >= l.date) ? r[0] : (l?.date ?? null);
      // the published latest value is exact; the timeline keeps 3 significant digits
      const value = l && l.date === date ? l.value : r && date === r[0] ? r[1] : null;
      return date && value != null ? { v, date, value, age: Math.floor((now.getTime() - Date.parse(`${date}T12:00:00Z`)) / 86400000) } : null;
    })
    .filter((x): x is NonNullable<typeof x> => !!x);
  const label: Record<string, string> = { pn_seriata: "Pseudo-nitzschia, larger", pn_delicatissima: "Pseudo-nitzschia, smaller", pDA: "Domoic acid (seawater)", chl_extracted: "Chlorophyll (extracted)" };
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      transition={{ type: "spring", stiffness: 380, damping: 32 }}
      className="absolute bottom-3 left-3 right-3 z-10 max-w-[340px] rounded-2xl border border-hairline bg-surface/95 p-3.5 shadow-[0_8px_28px_rgba(6,17,30,0.28)] backdrop-blur"
      data-testid="station-peek"
      role="dialog"
      aria-label={`${station.name}: latest measurements`}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="cw-kind text-measured">
          <Mark kind="measured" />
          {station.name} · {km < 10 ? km.toFixed(1) : Math.round(km)} km
        </p>
        <button type="button" onClick={onClose} aria-label="Close" className="-mr-1 -mt-1 grid h-8 w-8 place-items-center rounded-full text-ink-3 hover:text-ink" data-testid="station-peek-close">
          ×
        </button>
      </div>
      <ul className="mt-1.5 space-y-1 text-[14px]">
        {rows.length ? (
          rows.map((r) => (
            <li key={r.v} className="flex justify-between gap-3">
              <span className="text-ink-2">{label[r.v] ?? r.v}</span>
              <span className={`text-right tabular ${r.age > RECENT_DAYS ? "text-ink-3" : "font-medium text-ink"}`}>
                {fmtMeasured(r.v, r.value)} · {ageWords(r.age)}
              </span>
            </li>
          ))
        ) : (
          <li className="text-ink-2">No HAB measurement published for this station.</li>
        )}
      </ul>
      <Link href={`/bloom?station=${encodeURIComponent(station.station_id)}`} className="mt-2 inline-block text-[13px] font-medium text-accent hover:underline" data-testid="station-peek-history">
        Its full history on Bloom →
      </Link>
    </motion.div>
  );
}

/** What the map is drawing, in one line per mark (only the marks in view). */
export function MapKey({ focus, gradient }: { focus: string | null; gradient?: string | null }) {
  return (
    <div className="pointer-events-none absolute left-3 top-3 z-10 rounded-xl bg-[#06111e]/80 px-3 py-2 text-[12px] leading-[1.6] text-white/90 backdrop-blur" data-testid="coast-map-key">
      <p className="flex items-center gap-2">
        <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-[#4fd1ae]" /> Station sampled in the last {RECENT_DAYS} days
      </p>
      <p className="flex items-center gap-2">
        <span aria-hidden className="h-2.5 w-2.5 rounded-full border-2 border-[#4fd1ae]" /> Station, older samples only
      </p>
      {focus === "model" && (
        <>
          {gradient && (
            <p className="flex items-center gap-2">
              <span aria-hidden className="h-2.5 w-6 rounded-sm" style={{ background: gradient }} /> Model probability, 0–100%
            </p>
          )}
          <p className="flex items-center gap-2">
            <span aria-hidden className="h-0 w-3 border-t-2 border-dashed border-[#b58cf0]" /> 15 km around your waters
          </p>
          <p className="flex items-center gap-2">
            <span aria-hidden className="h-2.5 w-2.5 bg-[repeating-linear-gradient(45deg,#ffffff55_0_2px,transparent_2px_4px)]" /> No model value (often nearshore)
          </p>
        </>
      )}
      {focus === "satellite" && (
        <p className="flex items-center gap-2">
          <span aria-hidden className="h-2.5 w-6 rounded-sm" style={{ background: gradient ?? "#4fd1ae" }} /> Satellite chlorophyll (algae, not toxin)
        </p>
      )}
    </div>
  );
}
