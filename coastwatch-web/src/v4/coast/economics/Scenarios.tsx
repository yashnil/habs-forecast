"use client";

import { useRef, useState } from "react";
import type { Audience } from "@/lib/coast";
import { num } from "@/lib/tripEconomics";
import { compareScenarios, type FieldKey, type ScalarOutputKey } from "@/lib/tripEvaluation";
import { MAX_SCENARIOS, SHARE_PARAM, draftInputs, encodeShare, type Draft, type ScenarioRecord } from "@/lib/tripEconomicsStore";
import { toast } from "@/v4/ui/toast";
import { Action, Caps, TextIn } from "@/v4/coast/ui/Instrument";
import { notifyTrip, tripStore, useTripData } from "./useTripData";
import { econAudience, evaluateDraft, money } from "./model";
import { refsUsed } from "./draft";

/**
 * Saved scenarios (on this device only), a comparison that lists only the inputs that differ and each
 * result for both with its difference, the numbers-only share link, and the device's trip data.
 */

const CREW_WORDS: Record<string, string> = { none: "No hired crew", afterExpenses: "Share after expenses", ofGross: "Share of gross", dayRate: "Day rate" };

function inputLabel(k: FieldKey, d: Draft): string {
  const m = /^(catch|costs)\.(\d+)\.(\w+)$/.exec(k);
  if (m) {
    const i = Number(m[2]);
    if (m[1] === "catch") {
      const name = d.inputs.catch?.[i]?.species || `Catch ${i + 1}`;
      return m[3] === "lb" ? `${name}, lb` : m[3] === "usdPerLb" ? `${name}, $/lb` : `Catch line ${i + 1}`;
    }
    const name = d.inputs.costs?.[i]?.label || `Cost ${i + 1}`;
    return m[3] === "usd" ? `${name}, $` : m[3] === "shared" ? `${name}, shared` : `Cost line ${i + 1}`;
  }
  const L: Record<string, string> = {
    roundTripNm: "Round trip, nm",
    cruiseKt: "Cruising speed, kt",
    steamGph: "Steaming burn, gal/h",
    fishingHours: "Hours fishing",
    fishingGph: "Fishing burn, gal/h",
    fuelGal: "Fuel used, gal",
    fuelUsdPerGal: "Fuel price, $/gal",
    fuelShared: "Fuel comes off before shares",
    feesPctOfGross: "Fees, % of gross",
    feesUsdPerLb: "Fees, $/lb",
    "crew.kind": "Crew paid by",
    "crew.pct": "Crew share, %",
    "crew.crew": "Crew, people",
    "crew.usdPerDay": "Day rate, $",
    "crew.days": "Days paid",
    anglers: "People sharing the cost",
  };
  return L[k] ?? k;
}

const showVal = (v: number | string | boolean | null, k: FieldKey) => (v === null || v === "" ? "—" : typeof v === "boolean" ? (v ? "Yes" : "No") : k === "crew.kind" ? (CREW_WORDS[String(v)] ?? String(v)) : typeof v === "number" ? num(v, 3) : String(v));

const OUT_WORDS: Partial<Record<ScalarOutputKey, string>> = { gross: "Gross", tripCost: "Trip cost", crewPay: "Crew pay", netToBoat: "Net to boat", breakEvenCatch: "Break-even catch, lb", breakEvenPrice: "Break-even price, $/lb", perPerson: "Per person" };

function Compare({ a, b, audience }: { a: { label: string; draft: Draft }; b: { label: string; draft: Draft }; audience: Audience }) {
  const c = compareScenarios(draftInputs(a.draft), draftInputs(b.draft), { audience: econAudience(audience) });
  const outs = c.outputs.filter((o) => OUT_WORDS[o.output]);
  const fmtOut = (k: ScalarOutputKey, v: number | null, signed = false) => (v === null ? "—" : k === "breakEvenCatch" ? num(v, 0) : k === "breakEvenPrice" ? `$${v.toFixed(2)}` : money(v, signed));
  return (
    <table className="w-full table-fixed border-collapse text-[14px]" data-testid="econ-cmp-table">
      <thead>
        <tr className="text-left">
          <th scope="col" className="w-[38%] pb-2 font-normal">
            <span className="sr-only">Item</span>
          </th>
          <th scope="col" className="truncate pb-2 text-right font-normal">
            <Caps>A · {a.label}</Caps>
          </th>
          <th scope="col" className="truncate pb-2 text-right font-normal">
            <Caps>B · {b.label}</Caps>
          </th>
          <th scope="col" className="w-[18%] pb-2 text-right font-normal">
            <Caps>B − A</Caps>
          </th>
        </tr>
      </thead>
      <tbody>
        {c.inputs.length === 0 ? (
          <tr className="border-t border-n-line">
            <td colSpan={4} className="py-2 text-n-ink-2">
              Same inputs.
            </td>
          </tr>
        ) : (
          c.inputs.map((d) => (
            <tr key={d.field} className="border-t border-n-line" data-testid="econ-cmp-diff">
              <th scope="row" className="py-2 pr-2 text-left font-normal text-n-ink-2">
                {inputLabel(d.field, b.draft)}
              </th>
              <td className="py-2 text-right font-data">{showVal(d.a, d.field)}</td>
              <td className="py-2 text-right font-data">{showVal(d.b, d.field)}</td>
              <td />
            </tr>
          ))
        )}
        {outs.map((o, n) => (
          <tr key={o.output} className={`border-t ${n === 0 ? "border-n-line-strong" : "border-n-line"}`} data-testid="econ-cmp-out" data-output={o.output}>
            <th scope="row" className="py-2 pr-2 text-left font-medium text-n-ink">
              {OUT_WORDS[o.output]}
            </th>
            <td className="py-2 text-right font-data">{fmtOut(o.output, o.a, o.output === "netToBoat")}</td>
            <td className="py-2 text-right font-data">{fmtOut(o.output, o.b, o.output === "netToBoat")}</td>
            <td className={`py-2 text-right font-data ${o.delta !== null && o.delta < 0 ? "text-n-down" : "text-n-ink-2"}`} data-testid={`econ-cmp-delta-${o.output}`}>
              {o.delta === null ? "—" : o.delta === 0 ? "0" : o.output === "breakEvenCatch" ? `${o.delta > 0 ? "+" : "−"}${num(Math.abs(o.delta), 0)}` : o.output === "breakEvenPrice" ? `${o.delta > 0 ? "+" : "−"}$${Math.abs(o.delta).toFixed(2)}` : money(o.delta, true)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Scenarios({ place, species, audience, draft, scenarioId, setScenarioId, readOnly, onOpen }: { place: { key: string; name: string }; species: string; audience: Audience; draft: Draft; scenarioId: string | null; setScenarioId: (id: string | null) => void; readOnly: boolean; onOpen: (d: Draft) => void }) {
  const data = useTripData();
  const list = data?.scenarios ?? [];
  const current = list.find((s) => s.id === scenarioId) ?? null;
  const [name, setName] = useState("");
  const [a, setA] = useState<string>("current");
  const [b, setB] = useState<string>("");
  const say = (res: { ok: boolean; reason?: string }, ok: string) => toast(res.ok ? ok : res.reason === "full" ? `${MAX_SCENARIOS} scenarios kept: delete one first` : res.reason === "readOnly" ? "Read-only here" : "Not saved on this device");
  const save = (asNew: boolean) => {
    const res = tripStore().saveScenario({ id: asNew ? undefined : (scenarioId ?? undefined), name: name || (asNew && current ? `${current.name} (copy)` : current?.name) || `${place.name}, ${new Date().toLocaleDateString("en-US", { month: "short", day: "numeric" })}`, placeKey: place.key, species, draft, refs: refsUsed(draft) });
    notifyTrip();
    say(res, "Scenario saved on this device");
    if (res.ok && "id" in res && res.id) {
      setScenarioId(res.id);
      setName("");
    }
  };
  const opts = [{ value: "current", label: "This worksheet" }, ...list.map((s) => ({ value: s.id, label: s.name }))];
  const pick = (id: string): { label: string; draft: Draft } | null => {
    if (id === "current") return { label: "This worksheet", draft };
    const s = list.find((x) => x.id === id);
    return s ? { label: s.name, draft: s.draft } : null;
  };
  const A = pick(a);
  const B = pick(b);
  const sel = "nx-select nx-focus min-h-11 w-full rounded-[3px] border border-n-line-strong bg-n-surface pl-3 text-[15px] text-n-ink";
  return (
    <section aria-labelledby="econ-scen-h" className="flex flex-col gap-4" data-testid="econ-scenarios">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
        <h3 id="econ-scen-h" className="font-serif font-medium tracking-[-0.015em] leading-[1.1] text-[24px] text-n-ink">
          Scenarios
        </h3>
        <span className="text-[13px] text-n-ink-2">
          {list.length} of {MAX_SCENARIOS} kept on this device
        </span>
      </div>
      <div className="grid grid-cols-1 items-end gap-3 sm:grid-cols-[minmax(0,1fr)_auto_auto]">
        <TextIn label={current ? `Name (open: ${current.name})` : "Name this scenario"} value={name || current?.name || ""} onChange={setName} testId="econ-scen-name" />
        {!readOnly && (
          <>
            <Action primary onClick={() => save(false)} testId="econ-scen-save">
              {current ? "Save changes" : "Save scenario"}
            </Action>
            {current && (
              <Action onClick={() => save(true)} testId="econ-scen-dup">
                Save as new
              </Action>
            )}
          </>
        )}
      </div>
      {list.length > 0 && (
        <ul className="flex flex-col border-t border-n-line" data-testid="econ-scen-list">
          {list.map((s: ScenarioRecord) => {
            const ev = evaluateDraft(s.draft, audience);
            const head = ev.audience === "recreational" ? ev.tripCost : ev.netToBoat;
            return (
              <li key={s.id} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 border-b border-n-line py-2" data-testid="econ-scen-row" data-on={s.id === scenarioId || undefined}>
                <span className="min-w-0">
                  <span className="block truncate text-[15px] text-n-ink">
                    {s.name}
                    {s.id === scenarioId && <span className="ml-2 text-[12.5px] text-n-ink-2">open</span>}
                  </span>
                  <span className="block font-data text-[13px] text-n-ink-2">
                    {ev.audience === "recreational" ? "cost " : "net "}
                    {head.status === "value" ? money(head.value, ev.audience !== "recreational") : "not worked out"}
                    {s.placeKey !== place.key ? ` · another place` : ""}
                  </span>
                </span>
                <span className="flex flex-wrap justify-end gap-1">
                  <button
                    type="button"
                    className="nx-focus min-h-11 px-2 text-[13.5px] text-n-ink underline decoration-n-line-strong underline-offset-4 hover:decoration-n-ink"
                    onClick={() => {
                      setScenarioId(s.id);
                      setName("");
                      onOpen(s.draft);
                    }}
                    data-testid="econ-scen-open"
                  >
                    Open
                  </button>
                  <button type="button" className="nx-focus min-h-11 px-2 text-[13.5px] text-n-ink underline decoration-n-line-strong underline-offset-4 hover:decoration-n-ink" onClick={() => setB(s.id)} data-testid="econ-scen-compare">
                    Compare
                  </button>
                  {!readOnly && (
                    <button
                      type="button"
                      className="nx-focus min-h-11 px-2 text-[13.5px] text-n-ink-2 underline decoration-n-line-strong underline-offset-4 hover:text-n-ink"
                      onClick={() => {
                        const res = tripStore().deleteScenario(s.id);
                        notifyTrip();
                        say(res, "Scenario deleted");
                        if (s.id === scenarioId) setScenarioId(null);
                        if (s.id === b) setB("");
                      }}
                      data-testid="econ-scen-delete"
                    >
                      Delete
                    </button>
                  )}
                </span>
              </li>
            );
          })}
        </ul>
      )}
      {list.length > 0 && (
        <div className="flex flex-col gap-3" data-testid="econ-compare">
          <Caps as="h4">Compare two · only the inputs that differ</Caps>
          <div className="grid grid-cols-1 gap-3 min-[400px]:grid-cols-2">
            <label className="flex flex-col gap-1 text-[13.5px] text-n-ink-2">
              A
              <select value={a} onChange={(e) => setA(e.target.value)} className={sel} data-testid="econ-cmp-a">
                {opts.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-[13.5px] text-n-ink-2">
              B
              <select value={b} onChange={(e) => setB(e.target.value)} className={sel} data-testid="econ-cmp-b">
                <option value="">Choose…</option>
                {opts.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {A && B && <Compare a={A} b={B} audience={audience} />}
        </div>
      )}
    </section>
  );
}

export function ShareLink({ draft }: { draft: Draft }) {
  const [url, setUrl] = useState<string | null>(null);
  return (
    <div className="flex flex-col gap-2" data-testid="econ-share">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <Action
          testId="econ-share-copy"
          onClick={async () => {
            const u = `${window.location.origin}${window.location.pathname}${window.location.search}#${SHARE_PARAM}=${encodeShare(draft)}`;
            setUrl(u);
            try {
              await navigator.clipboard.writeText(u);
              toast("Link copied");
            } catch {
              // shown below to copy by hand
            }
          }}
        >
          Copy scenario link
        </Action>
        <span className="text-[13px] text-n-ink-2">Numbers only, no names. Only if you press it.</span>
      </div>
      {url && <input readOnly value={url} aria-label="Scenario link" onFocus={(e) => e.currentTarget.select()} className="nx-focus min-h-11 w-full rounded-[3px] border border-n-line bg-n-surface-2 px-3 font-data text-[13px] text-n-ink-2" data-testid="econ-share-url" />}
    </div>
  );
}

export function StoreStatus() {
  const data = useTripData();
  if (!data) return null;
  const s = data.status;
  const msgs: string[] = [];
  if (!s.available) msgs.push("This browser is not keeping data for this site: the worksheet works, but nothing is kept.");
  if (s.readOnly) msgs.push("Trip data here was saved by a newer version of Neritic. It is shown read-only so nothing is overwritten.");
  if (s.corrupt.length) msgs.push(`Some trip data on this device could not be read (${s.corrupt.join(", ")}). It is kept, and a copy is set aside before anything new is saved. Export it below if you need it.`);
  if (s.invalidRecords) msgs.push(`${s.invalidRecords} saved item${s.invalidRecords === 1 ? "" : "s"} could not be read and ${s.invalidRecords === 1 ? "is" : "are"} kept unchanged.`);
  if (!msgs.length) return null;
  return (
    <div className="flex flex-col gap-1 border-l-2 border-n-down pl-3 text-[14px] text-n-ink" role="status" data-testid="econ-store-status">
      {msgs.map((m) => (
        <p key={m}>{m}</p>
      ))}
    </div>
  );
}

export function TripDataPanel() {
  const [confirm, setConfirm] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const file = useRef<HTMLInputElement | null>(null);
  return (
    <details className="group border-t border-n-line pt-1" data-testid="econ-data">
      <summary className="nx-focus flex min-h-11 cursor-pointer list-none items-center gap-2 text-[14.5px] text-n-ink">
        Your trip data <span className="text-[13px] text-n-ink-2">· export, import, delete</span>
      </summary>
      <div className="flex flex-col gap-3 pb-2 pt-1">
        <p className="text-[14px] text-n-ink-2">Kept on this device only. Never sent to Neritic&rsquo;s server. Export a file to move it to another device.</p>
        <div className="flex flex-wrap gap-2">
          <Action
            testId="econ-export"
            onClick={() => {
              const blob = new Blob([tripStore().exportJson()], { type: "application/json" });
              const a = document.createElement("a");
              a.href = URL.createObjectURL(blob);
              a.download = `neritic-trips-${new Date().toISOString().slice(0, 10)}.json`;
              a.click();
              setTimeout(() => URL.revokeObjectURL(a.href), 1000);
            }}
          >
            Export (JSON)
          </Action>
          <Action testId="econ-import" onClick={() => file.current?.click()}>
            Import
          </Action>
          <input
            ref={file}
            type="file"
            accept="application/json,.json"
            className="sr-only"
            tabIndex={-1}
            aria-hidden
            data-testid="econ-import-file"
            onChange={async (e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (!f) return;
              const res = tripStore().importJson(await f.text());
              notifyTrip();
              setMsg(res.ok ? `Imported: ${res.added} added, ${res.updated} updated${res.skipped ? `, ${res.skipped} skipped (limit reached or unreadable)` : ""}.` : res.reason === "newer" ? "That file is from a newer version of Neritic." : res.reason === "readOnly" ? "Read-only here." : "That file is not a Neritic trip export.");
            }}
          />
          {confirm ? (
            <>
              <Action
                primary
                testId="econ-delete-confirm"
                onClick={() => {
                  const res = tripStore().deleteAll();
                  notifyTrip();
                  setConfirm(false);
                  setMsg(res.ok ? "All trip data on this device was deleted." : "Could not delete here.");
                }}
              >
                Delete everything
              </Action>
              <Action testId="econ-delete-cancel" onClick={() => setConfirm(false)}>
                Keep it
              </Action>
            </>
          ) : (
            <Action testId="econ-delete-all" onClick={() => setConfirm(true)}>
              Delete all trip data
            </Action>
          )}
        </div>
        {msg && (
          <p className="text-[14px] text-n-ink-2" role="status" data-testid="econ-data-msg">
            {msg}
          </p>
        )}
      </div>
    </details>
  );
}
