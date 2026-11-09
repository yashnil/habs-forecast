"use client";

import { useEffect, useId, useState } from "react";
import { PILOT_PORTS } from "@/pilot/contracts";
import { usePilotAccount } from "./context";
import { FISHERIES, placeLabel } from "./local";
import { deleteTrip, readTrips, saveTrip, type OutcomeBand, type TripLog } from "./records";
import { sendTrip } from "./sync";

const OUTCOMES: { id: OutcomeBand; label: string }[] = [
  { id: "skipped", label: "Didn't go" },
  { id: "poor", label: "Poor" },
  { id: "fair", label: "Fair" },
  { id: "good", label: "Good" },
];

const pacificToday = () => new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles" }).format(new Date());

/**
 * A deliberately minimal, optional trip log: date, port, fishery, how it went (a band, never catch
 * numbers or locations), what the water was like, and whether Neritic helped. Kept on this device;
 * also sent to the signed-in account.
 */
export function TripLogForm({ defaultPlace }: { defaultPlace?: string | null }) {
  const { session } = usePilotAccount();
  const id = useId();
  const [trips, setTrips] = useState<TripLog[]>([]);
  const [date, setDate] = useState("");
  const [place, setPlace] = useState(defaultPlace ?? PILOT_PORTS[0].place_id);
  const [fishery, setFishery] = useState("");
  const [outcome, setOutcome] = useState<OutcomeBand | "">("");
  const [observed, setObserved] = useState("");
  const [useful, setUseful] = useState<"" | "yes" | "no">("");
  const [msg, setMsg] = useState("");
  useEffect(() => {
    setTrips(readTrips());
    setDate(pacificToday());
  }, []);
  useEffect(() => {
    if (defaultPlace) setPlace(defaultPlace);
  }, [defaultPlace]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const t: TripLog = {
      id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
      trip_date: date,
      place_id: place,
      fishery: fishery || null,
      outcome_band: outcome || null,
      observed_conditions: observed.trim() ? observed.trim().slice(0, 1000) : null,
      recommendation_useful: useful === "" ? null : useful === "yes",
      at: new Date().toISOString(),
    };
    const kept = saveTrip(t);
    setTrips(readTrips());
    setObserved("");
    setOutcome("");
    setUseful("");
    let where = kept ? "on this device" : "for this visit only (this browser is not keeping data)";
    if (session.client && session.status === "signed_in") {
      const r = await sendTrip(session.client, t);
      if (r.ok) where = "to your account";
    }
    setMsg(`Trip saved ${where}.`);
  }

  const field = "pilot-field nx-focus";
  return (
    <div data-testid="trip-log">
      <form onSubmit={submit} className="grid gap-3 sm:grid-cols-2">
        <div>
          <label htmlFor={`${id}-date`} className="pilot-label">Date</label>
          <input id={`${id}-date`} type="date" required value={date} onChange={(e) => setDate(e.target.value)} className={field} />
        </div>
        <div>
          <label htmlFor={`${id}-place`} className="pilot-label">Port</label>
          <select id={`${id}-place`} value={place} onChange={(e) => setPlace(e.target.value)} className={field}>
            {PILOT_PORTS.map((p) => (
              <option key={p.place_id} value={p.place_id}>{p.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor={`${id}-fishery`} className="pilot-label">Fishery (optional)</label>
          <select id={`${id}-fishery`} value={fishery} onChange={(e) => setFishery(e.target.value)} className={field}>
            <option value="">—</option>
            {FISHERIES.map((f) => (
              <option key={f.id} value={f.id}>{f.label}</option>
            ))}
          </select>
        </div>
        <fieldset>
          <legend className="pilot-label">How did it go? (optional)</legend>
          <div className="flex flex-wrap gap-1.5">
            {OUTCOMES.map((o) => (
              <label key={o.id} className={`pilot-choice ${outcome === o.id ? "on" : ""}`}>
                <input type="radio" name={`${id}-outcome`} value={o.id} checked={outcome === o.id} onChange={() => setOutcome(o.id)} className="sr-only" />
                {o.label}
              </label>
            ))}
          </div>
        </fieldset>
        <div className="sm:col-span-2">
          <label htmlFor={`${id}-obs`} className="pilot-label">What was the water like? (optional)</label>
          <input id={`${id}-obs`} value={observed} maxLength={1000} onChange={(e) => setObserved(e.target.value)} placeholder="e.g. brown water off the jetty, 6 ft swell" className={field} />
        </div>
        <fieldset className="sm:col-span-2">
          <legend className="pilot-label">Did Neritic help with this trip? (optional)</legend>
          <div className="flex gap-1.5">
            {(["yes", "no"] as const).map((v) => (
              <label key={v} className={`pilot-choice ${useful === v ? "on" : ""}`}>
                <input type="radio" name={`${id}-useful`} value={v} checked={useful === v} onChange={() => setUseful(v)} className="sr-only" />
                {v === "yes" ? "Yes" : "No"}
              </label>
            ))}
          </div>
        </fieldset>
        <div className="sm:col-span-2">
          <button type="submit" className="today-btn primary nx-press nx-focus !min-h-12 px-5" data-testid="trip-save">
            Save trip
          </button>
          <span role="status" className="ml-3 text-[13.5px] text-n-ink-2" data-testid="trip-status">{msg}</span>
        </div>
      </form>
      {trips.length > 0 && (
        <ul className="mt-5 border-t border-n-line" data-testid="trip-list">
          {trips.slice(0, 10).map((t) => (
            <li key={t.id} className="flex items-center justify-between gap-3 border-b border-n-line py-1.5 text-[14.5px] text-n-ink-2">
              <span>
                <span className="font-medium text-n-ink">{t.trip_date}</span> · {placeLabel(t.place_id)}
                {t.outcome_band ? ` · ${OUTCOMES.find((o) => o.id === t.outcome_band)?.label}` : ""}
              </span>
              <button
                type="button"
                onClick={() => {
                  deleteTrip(t.id);
                  setTrips(readTrips());
                }}
                className="pilot-link nx-focus min-h-11 px-2 text-[13.5px]"
                aria-label={`Delete trip on ${t.trip_date} at ${placeLabel(t.place_id)}`}
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
