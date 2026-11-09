import type { Metadata } from "next";
import { loadToday, fixedNow } from "@/pilot/today/load";
import { parseFisheries } from "@/pilot/facts";
import { readiness } from "@/pilot/readiness";
import { ReadinessProvider } from "@/v4/readiness/store";
import { DataUnavailable } from "@/v4/shell/DataUnavailable";
import { TodayView } from "@/pilot/today/TodayView";
import { PortPicker } from "@/pilot/today/PortPicker";
import { loadTrust } from "@/lib/data";
import { localNotices } from "@/lib/applicability";
import { pilotAccountConfig } from "@/pilot/account/config";
import { PilotAccountProvider } from "@/pilot/account/context";

export const dynamic = "force-dynamic";

type SP = Promise<Record<string, string | string[] | undefined>>;
const one = (v: string | string[] | undefined) => (typeof v === "string" ? v : null);

export async function generateMetadata({ searchParams }: { searchParams: SP }): Promise<Metadata> {
  const place = one((await searchParams).place);
  return { title: place ? "Today at your port" : "Today: choose your port" };
}

/**
 * Today at your port (Neritic private pilot): one page per port, answering in under a minute what is
 * open, restricted or delayed, what the latest toxin tests say, what the sea is doing, the bloom
 * outlook, the whale rules and what changed (src/pilot/facts.ts builds every statement).
 * `?place=port-592` opens a port; with none, the reader's last or home port opens, else a picker.
 * `?f=dungeness_commercial,groundfish` sets the fisheries shown (otherwise the reader's saved list).
 */
export default async function TodayPage({ searchParams }: { searchParams: SP }) {
  const sp = await searchParams;
  const [load, base] = await Promise.all([loadToday(one(sp.place)), loadTrust()]);
  if (!load.ok) return <DataUnavailable error={load.error} />;
  // the trust pill counts the notices that cover this port's waters
  const trust = load.place ? { ...base, local: localNotices(base.official, load.place.key, load.place.name, load.place.lat, load.place.lon) } : base;
  return (
    // the same account context as the other pilot pages: the bell and "Was this useful?" follow the signed-in account
    <PilotAccountProvider config={pilotAccountConfig()}>
      <ReadinessProvider>
        {load.place && load.inputs ? (
          <TodayView key={load.place.key} inputs={load.inputs} linkFisheries={parseFisheries(one(sp.f))} readiness={{ alerts: readiness("alerts"), ask: readiness("ask"), sms: readiness("sms") }} fixture={load.fixture} trust={trust} />
        ) : (
          <PortPicker ports={load.ports} requested={load.requested} pick={one(sp.pick) === "1"} now={fixedNow().toISOString()} trust={trust} />
        )}
      </ReadinessProvider>
    </PilotAccountProvider>
  );
}
